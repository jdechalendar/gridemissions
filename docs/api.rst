.. _api:

API reference
=============
This page gives an overview of all public ``gridemissions`` objects.

.. currentmodule:: gridemissions

Data structure
--------------
The main abstraction we use for holding data is :class:`GraphData`

.. autosummary::
  :toctree: generated/

  GraphData
  read_csv
  GraphData.to_csv

Accessing data

.. autosummary::
  :toctree: generated/

  GraphData.get_data
  GraphData.get_cols
  GraphData.has_field

There are several helper methods to check data consistency.

.. autosummary::
  :toctree: generated/

  GraphData.check_all
  GraphData.check_antisymmetric
  GraphData.check_balance
  GraphData.check_generation_by_source
  GraphData.check_interchange
  GraphData.check_nans
  GraphData.check_positive

Data cleaning
-------------
.. autosummary::
  :toctree: generated/

  BasicCleaner
  BasicCleaner.process
  RollingCleaner
  RollingCleaner.process

Data reconciliation
-------------------
.. autosummary::
  :toctree: generated/

   CvxCleaner
   CvxCleaner.process

Consumption-based emissions
---------------------------
.. autosummary::
  :toctree: generated/

  EmissionsCalc
  EmissionsCalc.process
