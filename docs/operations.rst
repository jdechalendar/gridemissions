.. _operations:

.. currentmodule:: gridemissions

Operations
==========
Two main steps are needed to create the datasets for the the visualization at `energy.stanford.edu/gridemissions`_.

#. Reconciliation of electric system operating data
#. Computing consumption-based emissions


Data cleaning and reconciliation of electric system operating data
------------------------------------------------------------------
Raw electric grid data typically have errors and inconsistencies, but many applications require "clean" data. For example, computing consumption-based emissions (see following sections). ``gridemissions`` implements the following three data cleaning operations that were initially described in de Chalendar and Benson (2021).

#. Basic data cleaning using ad-hoc information for the US balancing areas, implemented by :class:`BasicCleaner`
#. Rolling window cleaning, implemented by :class:`RollingCleaner`
#. Physics-based data reconciliation, implemented by :class:`CvxCleaner`. This operations uses an optimization-based algorithms to minimize the data adjustments required to satisfy energy conservation equations. Weights can be used to control the relative magnitude of adjustments in different data channels.

Two important notes:

#. We use these operations sequentially to produce the dataset that we release, but they can be used independently.
#. A user with access to more reliable information on the input data could manually pre-process data, effectively replacing the heuristics we use in steps 1 and 2, and then use the algorithm in 3 to ensure that the final dataset is internally consistent. In that case, the weights that are passed to :class:`CvxCleaner` can be used to place a high penalty on deviating from user-supplied inputs.

Consumption-based emissions
---------------------------
This operation was initially described in de Chalendar et al. (2019) and is implemented by :class:`EmissionsCalc`.

Electric grid data on production, consumption and exchanges, along with the emissions associated with electricity production, are used to compute the emissions embodied in electricity **consumption**. By default, we are using IPCC Life-Cycle Assessment emissions factors to compute the emissions associated with generating electricity from different sources, so the CO2 data we release are in units of CO2-eq. If you wish to use different emissions factors, or factors for other quantities (e.g. SO2, NOx, PM2.5, or H2O), you can use the tools in this package to generate corresponding consumption-based data.

References
----------
#. de Chalendar and Benson (2021), "A Physics-informed data reconciliation framework for real-time electricity and emissions tracking", Applied Energy 304, 117761; DOI: 10.1016/j.apenergy.2021.117761.
#. de Chalendar, Taggart, and Benson (2019), "Tracking emissions in the US electricity system", Proceedings of the National Academy of Sciences, 116 (51) 25497-25502; DOI: 10.1073/pnas.1912950116


.. _energy.stanford.edu/gridemissions: https://energy.stanford.edu/gridemissions
