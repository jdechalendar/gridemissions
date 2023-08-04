"""
Tools to interact with the AMPD ftp at newftp.epa.gov
"""
import os
from os.path import join
import pandas as pd
from ftplib import FTP
import logging


class AMPD_download(object):
    def __init__(self, pathOut):
        self.ftp = FTP("newftp.epa.gov")
        os.makedirs(pathOut, exist_ok=True)
        self.pathOut = pathOut
        self.logger = logging.getLogger("scraper")
        self.localStatus = {}
        self.remoteStatus = {}

    def login(self):
        self.ftp.login()

    def close(self):
        self.ftp.close()

    def getStatus(self, year=None):
        self.ftp.cwd("/dmdnload/emissions/hourly/monthly")
        self.years = self.ftp.nlst()

        if year is not None:
            years = [str(year)]
        else:
            years = self.years

        for y in years:
            self.logger.info("Retrieving ftp status for %s" % y)
            self.localStatus[y] = self.getLocalStatus(y)
            self.remoteStatus[y] = self.getRemoteStatus(y)
            self.logger.info("%s has %d files" % (y, len(self.remoteStatus[y])))

    def getLocalStatus(self, year):
        """
        Get status of AMPD dataset on local filesystem:
        What files are available.
        TODO: store and reload what the modification date was the last
        time we downloaded stuff.
        """
        folderYear = join(self.pathOut, year)
        os.makedirs(folderYear, exist_ok=True)
        return pd.DataFrame([[f] for f in os.listdir(folderYear)], columns=["fileNm"])

    def getRemoteStatus(self, year):
        """
        Get status of AMPD dataset on EPA ftp:
        What files are available, what modification date.
        """
        self.ftp.cwd("/dmdnload/emissions/hourly/monthly/%s" % year)
        fileNms = self.ftp.nlst()

        fileNms = [(self.ftp.sendcmd("MDTM %s" % f), f) for f in fileNms]
        df = pd.DataFrame(fileNms, columns=["MDTM", "fileNm"])
        df.MDTM = df.MDTM.apply(
            lambda x: pd.to_datetime(x.split()[-1], format="%Y%m%d%H%M%S.%f")
        )
        return df

    def download(self, year):
        """
        Download a year's worth of data, skipping files that are already there.
        """
        year = str(year)
        self.logger.info("Downloading data for %s" % year)
        folderYear = join(self.pathOut, year)

        self.ftp.cwd("/dmdnload/emissions/hourly/monthly/%s" % year)

        cnt_downloaded = 0
        for irow, row in self.remoteStatus[year].iterrows():
            # Check to see if we already have this
            local = self.localStatus[year][self.localStatus[year].fileNm == row.fileNm]

            if len(local) == 0:
                download = True
            elif len(local) == 1:
                download = False
            else:
                logging.debug(local)
                raise ValueError("There are two fileNms that match %s" % row)

            if download:
                try:
                    self.logger.debug("Downloading %s ..." % row.fileNm)
                    with open(join(folderYear, row.fileNm), "wb") as f:
                        self.ftp.retrbinary("RETR %s" % row.fileNm, f.write)
                    cnt_downloaded += 1
                except TimeoutError:  # TODO add known errors here
                    return "timedout"
            else:
                self.logger.debug("Skipping %s ..." % row.fileNm)
        self.logger.info(
            "Downloaded  %d/%d files for %s"
            % (cnt_downloaded, len(self.remoteStatus[year]), year)
        )
        return "done"


class AMPDRaw(object):
    """
    Simple class to handle AMPD data.
    """

    def __init__(self, fileNms):
        self.logger = logging.getLogger("load")
        if isinstance(fileNms, str):
            fileNms = [fileNms]
        self.fileNms = fileNms

    def parse(self, cols=None):
        self.df = pd.concat([self._parse_file(f, cols) for f in self.fileNms])

    def _parse_file(self, fileNm, cols=None):
        """
        Parse one file downloaded from the EPA AMPD ftp server
            ftp://newftp.epa.gov/DMDnLoad/emissions/hourly/monthly/

        1. Create a datetime column.
        2. Change units to metric
        3. Select only columns we need
        """
        col_name_map = {
            "CO2_MASS": "CO2_MASS (tons)",
            "CO2_RATE": "CO2_RATE (tons/mmBtu)",
            "GLOAD": "GLOAD (MW)",
            "HEAT_INPUT": "HEAT_INPUT (mmBtu)",
            "NOX_MASS": "NOX_MASS (lbs)",
            "SLOAD": "SLOAD (1000lb/hr)",
            "SLOAD (1000 lbs)": "SLOAD (1000lb/hr)",
            "SO2_MASS": "SO2_MASS (lbs)",
            "SO2_RATE": "SO2_RATE (lbs/mmBtu)",
            "NOX_RATE": "NOX_RATE (lbs/mmBtu)",
        }
        us_to_si_tons = 0.9071847
        lbs_to_si_tons = 0.000453592
        mmbtu_to_mwh = 0.29329722222222

        # Read data
        self.logger.info("Loading %s" % fileNm)
        df_tmp = pd.read_csv(fileNm, compression="zip", low_memory=False)

        # Rename columns
        df_tmp.rename(col_name_map, axis="columns", inplace=True)

        # 1. Parse timestamp (local time)
        df_tmp.loc[:, "OP_DATE_TIME"] = pd.to_datetime(
            df_tmp["OP_DATE"] + "-" + df_tmp["OP_HOUR"].astype(str),
            format="%m-%d-%Y-%H",
        )
        #        df_tmp.loc[:, 'YEAR'] = df_tmp.loc[:, 'OP_DATE_TIME'].dt.year.astype(
        #            int)
        #        df_tmp.loc[:, 'MONTH'] = df_tmp.loc[:, 'OP_DATE_TIME'].dt.month.astype(
        #            int)

        # 2. Change units to metric
        df_tmp.loc[:, "CO2"] = df_tmp.loc[:, "CO2_MASS (tons)"] * us_to_si_tons
        df_tmp.loc[:, "SO2"] = df_tmp.loc[:, "SO2_MASS (lbs)"] * lbs_to_si_tons
        df_tmp.loc[:, "NOX"] = df_tmp.loc[:, "NOX_MASS (lbs)"] * lbs_to_si_tons
        df_tmp.loc[:, "HEAT_INPUT"] = df_tmp.loc[:, "HEAT_INPUT (mmBtu)"] * mmbtu_to_mwh
        df_tmp.loc[:, "GLOAD"] = df_tmp.loc[:, "GLOAD (MW)"]

        # 3. Keep as little info as possible to save space
        keep_cols = [
            "ORISPL_CODE",
            "UNITID",
            "OP_DATE_TIME",
            "OP_TIME",
            "GLOAD",
            "SO2",
            "NOX",
            "CO2",
            "HEAT_INPUT",
        ]

        if cols is None:
            return df_tmp.loc[:, keep_cols]
        elif cols == "all":
            return df_tmp
        else:
            return df_tmp.loc[:, cols]


def extract_state(x):
    """
    Combine all the files for a given state and year.
    """
    path_name, path_name_out, year, state = x
    fileNms = [
        join(path_name, f) for f in os.listdir(path_name) if f.startswith(year + state)
    ]
    logger = logging.getLogger("load")
    logger.info("Reading %d files for %s (%s)" % (len(fileNms), state, year))

    ampd = AMPDRaw(fileNms)
    ampd.parse()
    ampd.df.to_csv(join(path_name_out, "%s%s.csv" % (state, year)), index=False)


if __name__ == "__main__":
    from gridemissions import config

    # Download data for 2019
    DATA_PATH = config["DATA_PATH"]
    count_timed_out = 0
    status = "timedout"
    y = 2019
    while status == "timedout":
        ampd = AMPD_download(os.path.join(DATA_PATH, "raw", "AMPD"))
        ampd.login()
        ampd.getStatus(y)
        status = ampd.download(y)
        if status == "timed_out":
            count_timed_out += 1
        if count_timed_out > 5:
            raise ValueError("Too many timeouts")
        ampd.close()


#     # Extract data for California
#     state = "tx"
#     for year in ["2015", "2016", "2017", "2018", "2019"]:
#         path_in = join(config['DATA_PATH'], "raw", "AMPD", year)
#         path_out = join(config['DATA_PATH'], "analysis", "AMPD")
#         os.makedirs(path_out, exist_ok=True)
#         extract_state((path_in, path_out, year, state))
