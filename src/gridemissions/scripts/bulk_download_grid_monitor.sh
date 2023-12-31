mkdir -p $DATA_PATH/EIA_Grid_Monitor/downloads
cd $DATA_PATH/EIA_Grid_Monitor/downloads
echo "Downloading files to $PWD";

base_eia_url="https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
years=(2019 2020 2021 2022 2023)
heads=(EIA930_BALANCE_ EIA930_INTERCHANGE_)
tails=(_Jul_Dec.csv _Jan_Jun.csv)
for y in "${years[@]}"
do
    for head in "${heads[@]}"
    do
        for tail in "${tails[@]}"
        do
            wget $base_eia_url$head$y$tail;
        done
    done
done

# Only get the second part of the year for 2018
wget $base_eia_url"EIA930_BALANCE_2018_Jul_Dec.csv";
wget $base_eia_url"EIA930_INTERCHANGE_2018_Jul_Dec.csv";
