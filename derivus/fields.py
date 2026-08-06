from . import instruments
from .schema import emit_instrument

# the Instrument store is a VIEW of the per-class declarations, not a second copy
_types, _sections = emit_instrument(instruments)

# object list defaults
default = {
    'Integer': 0,
    'Float': 0.0,
    'Percent': 0.0,
    'Text': '',
    'Flot': '[{"label":"None", "data":[[0.0,0.0]]}]',
    'Surface': '[[0.0,1.0], [1.0,0.0]]',
    'Space': '{"0.0":[[0.0,0.0],[0.0,0.0]]}',
    'DateList': 'null',
    'CreditSupportList': '[[0,1]]',
    'DatePicker': ''
}

num_format = {
    'float': {'pattern': '0.000'},
    'int': {'pattern': '0.'},
    'percent': {'pattern': '0.00 %'},
    'currency': {'pattern': '0,0.00'}
}

# FXVol/EquityPriceVol/CommodityPriceVol were three declarations of one thing. What varies is the
# SUBTYPE (Surface_Type, Moneyness_Rule), not the asset class, so there is one type and one list.
_VOLATILITY_GRID = ["Surface_Type", "Surface", "Moneyness_Rule", "Delta_Surface", "ATM_Ref", "ATM_Vol",
                    "a", "b", "s", "L", "R", "C", "D", "lam", "rho", "m", "sigma", "Currency"]

# this whole thing could be stored as a json file . . .
mapping = {
    'Calibration': {
        'fields': {
            'MLE_Parameters': {'widget': 'Container', 'description': 'MLE Parameters',
                               'value': {"Data_Retrieval_Parameters":
                                             {"Start_Date": "", "End_Date": "", "Length": "", "Frequency": "1d",
                                              "Calendar": "", "Business_Days_In_Year": 252,
                                              "Diagnostics_Error_Level": "Info", "Data_Cleaning_Methods": "",
                                              "Horizon": ""},
                                         "Min_Tenor": "3M", "Reversion_Speed_Lower_Bound": 0.1,
                                         "Reversion_Speed_Upper_Bound": 4.0, "Yield_Volatility_Upper_Bound": "",
                                         "Exact_Solution_Optimisation_Parameters": {
                                             "Max_Iterations": 1000, "Fractional_Tolerance": 0.00000001,
                                             "Downhill_Simplex_Scale": 0.005}
                                         },
                               'sub_fields': ['Data_Retrieval_Parameters', 'Min_Tenor', 'Reversion_Speed_Fixed',
                                              'Reversion_Speed_Lower_Bound', 'Reversion_Speed_Upper_Bound',
                                              'Yield_Volatility_Upper_Bound',
                                              'Exact_Solution_Optimisation_Parameters']
                               },
            'Data_Retrieval_Parameters': {'widget': 'Container', 'description': 'Data Retrieval Parameters',
                                          'value': {"Start_Date": "", "End_Date": "", "Length": "", "Frequency": "1d",
                                                    "Calendar": "", "Business_Days_In_Year": 252,
                                                    "Diagnostics_Error_Level": "Info", "Data_Cleaning_Methods": "",
                                                    "Horizon": ""},
                                          'sub_fields': ['Start_Date', 'End_Date', 'Length', 'Frequency', 'Calendar',
                                                         'Business_Days_In_Year', 'Diagnostics_Error_Level',
                                                         'Data_Cleaning_Methods', 'Horizon']},
            'Exact_Solution_Optimisation_Parameters': {'widget': 'Container',
                                                       'description': 'Exact Solution Optimisation Parameters',
                                                       'value': {'Max_Iterations': 1000,
                                                                 'Fractional_Tolerance': 0.00000001,
                                                                 'Downhill_Simplex_Scale': 0.005},
                                                       'sub_fields': ['Max_Iterations', 'Fractional_Tolerance',
                                                                      'Downhill_Simplex_Scale']},
            'Max_Iterations': {'widget': 'Integer', 'description': 'Max Iterations', 'value': 1000},
            'Fractional_Tolerance': {'widget': 'Float', 'description': 'Fractional Tolerance', 'value': 0.00000001},
            'Downhill_Simplex_Scale': {'widget': 'Float', 'description': 'Downhill Simplex Scale', 'value': 0.005},
            'Start_Date': {'widget': 'DatePicker', 'description': 'Start Date', 'value': default['DatePicker']},
            'End_Date': {'widget': 'DatePicker', 'description': 'End Date', 'value': default['DatePicker']},
            'Frequency': {'widget': 'Text', 'description': 'Frequency', 'value': '1d', 'obj': 'Period'},
            'Length': {'widget': 'Text', 'description': 'Length', 'value': ''},
            'Calendar': {'widget': 'Text', 'description': 'Calendar', 'value': ''},
            'Horizon': {'widget': 'Text', 'description': 'Horizon', 'value': ''},
            'Diagnostics_Error_Level': {'widget': 'Dropdown', 'description': 'Diagnostics Error Level', 'value': 'Info',
                                        'values': ['None', 'Info', 'Warning', 'Error']},
            'Calibration_Method': {'widget': 'Dropdown', 'description': 'Calibration Method', 'value': 'MLE',
                                   'values': ['MLE', 'Pre_Computed_Statistics']},
            'Data_Cleaning_Methods': {'widget': 'Text', 'description': 'Data Cleaning Methods', 'value': ''},
            'Business_Days_In_Year': {'widget': 'Integer', 'description': 'Business Days In Year', 'value': 252},
            'Min_Tenor': {'widget': 'Text', 'description': 'Min Tenor', 'value': '3M', 'obj': 'Period'},
            'Reversion_Speed_Fixed': {'widget': 'Text', 'description': 'Reversion Speed Fixed', 'value': ''},
            'Reversion_Speed_Lower_Bound': {'widget': 'Float', 'description': 'Reversion Speed Lower Bound',
                                            'value': 0.1},
            'Reversion_Speed_Upper_Bound': {'widget': 'Float', 'description': 'Reversion Speed Upper Bound',
                                            'value': 3.0},
            'Yield_Volatility_Upper_Bound': {'widget': 'Text', 'description': 'Yield Volatility Upper Bound',
                                             'value': ''},
            'Number_PCA_Factors': {'name': 'Number_Of_PCA_Factors', 'widget': 'Integer', 'description': 'Number Of PCA Factors', 'value': 3},
            'Distribution_Type': {'widget': 'Dropdown', 'description': 'Distribution Type', 'value': 'Lognormal',
                                  'values': ['Lognormal', 'Normal']},
            'Use_Pre_Computed_Statistics': {'widget': 'Dropdown', 'description': 'Use Pre Computed Statistics',
                                            'value': 'No', 'values': ['Yes', 'No']},
            'Matrix_Type': {'widget': 'Dropdown', 'description': 'Matrix Type', 'value': 'Correlation',
                            'values': ['Correlation', 'Covariance']},
            'Rate_Drift_Model': {'widget': 'Dropdown', 'description': 'Rate Drift Model', 'value': 'Drift_To_Forward',
                                 'values': ['Drift_To_Forward', 'Drift_To_Blend']}
        },
        'types': {
            'PCAInterestRateModel': ['Calibration_Method', 'Number_PCA_Factors', 'Distribution_Type', 'Matrix_Type',
                                     'Rate_Drift_Model', 'MLE_Parameters'],
            'GBMAssetPriceModel': ['Use_Pre_Computed_Statistics', 'Data_Retrieval_Parameters']
        }
    },

    'Calculation': {

        'fields': {
            'Base_Date': {'widget': 'DatePicker', 'description': 'Base Date', 'value': default['DatePicker']},
            'Calculate': {'widget': 'Dropdown', 'description': 'Calculate', 'value': 'No', 'values': ['Yes', 'No']},
            'Counterparty': {'widget': 'Text', 'description': 'Counterparty', 'value': ''},
            'Collateral_Curve': {'widget': 'Text', 'description': 'Collateral Curve', 'value': ''},
            'Funding_Curve': {'widget': 'Text', 'description': 'Funding Curve', 'value': ''},
            'Risk_Free_Curve': {'widget': 'Text', 'description': 'Risk Free Curve', 'value': ''},
            'Funding_Cost_Interest_Curve': {'widget': 'Text', 'description': 'Funding Cost Interest Curve', 'value': ''},
            'Funding_Benefit_Interest_Curve': {'widget': 'Text', 'description': 'Funding Benefit Interest Curve', 'value': ''},
            'Collateral_Spread': {'widget': 'Integer', 'description': 'Collateral Spread', 'value': 0},
            'Funding_Spread': {'widget': 'Integer', 'description': 'Funding Spread', 'value': 0},
            'Bank': {'widget': 'Text', 'description': 'Bank', 'value': ''},
            'Deflate_Stochastically': {'widget': 'Dropdown', 'description': 'Deflate Stochastically', 'value': 'Yes',
                                       'values': ['Yes', 'No']},
            'Stochastic_Hazard_Rates': {'widget': 'Dropdown', 'description': 'Stochastic Hazard Rates', 'value': 'No',
                                        'values': ['Yes', 'No']},
            'Stochastic_Funding': {'widget': 'Dropdown', 'description': 'Stochastic Funding', 'value': 'No',
                                        'values': ['Yes', 'No']},
            'Gradient': {'widget': 'Dropdown', 'description': 'Gradient', 'value': 'No', 'values': ['Yes', 'No']},
            'Greeks': {'widget': 'Dropdown', 'description': 'Greeks', 'value': 'No', 'values': ['First', 'No']},
            'Antithetic': {'widget': 'Dropdown', 'description': 'Antithetic', 'value': 'No', 'values': ['Yes', 'No']},            
            'Base_Time_Grid': {'widget': 'Text', 'description': 'Base Time Grid',
                               'value': '0d 2d 1w(1w) 3m(1m) 2y(3m)'},
            'Dynamic_Scenario_Dates': {'widget': 'Dropdown', 'description': 'Dynamic Scenario Dates',
                                       'value': 'Yes', 'values': ['Yes', 'No']},
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': 'ZAR'},
            'Percentile': {'widget': 'Text', 'description': 'Percentile', 'value': '95'},
            'Simulation_Batches': {'widget': 'Integer', 'description': 'Simulation Batches', 'value': 1},
            'MCMC_Simulations': {'widget': 'Integer', 'description': 'MCMC Simulations', 'value': 2048},
            'Batch_Size': {'widget': 'Integer', 'description': 'Batch Size', 'value': 1024},
            'Random_Seed': {'widget': 'Integer', 'description': 'Random Seed', 'value': 5120},
            'Calc_Scenarios': {'widget': 'Dropdown', 'description': 'Calc Scenarios', 'value': 'No',
                               'values': ['At_Percentile', 'All', 'No']},
            'Deflation_Interest_Rate': {'widget': 'Text', 'description': 'Deflation Interest Rate',
                                        'value': 'ZAR-SWAP'},
            'Credit_Valuation_Adjustment': {'widget': 'Container', 'description': 'Credit Valuation Adjustment',
                                            'value': {"Calculate": "No", "Counterparty": "", "Bank": "",
                                                      "Deflate_Stochastically": "Yes", "Stochastic_Hazard_Rates": "No",
                                                      "Gradient": "No"},
                                            'sub_fields': ['Calculate', 'Counterparty', 'Bank',
                                                           'Deflate_Stochastically', 'Stochastic_Hazard_Rates',
                                                           'Gradient']},
            'Funding_Valuation_Adjustment': {'widget': 'Container', 'description': 'Funding Valuation Adjustment',
                                             'value': {"Calculate": "No", "Counterparty": "", "Bank": "",
                                                       "Risk_Free_Curve": "", "Funding_Cost_Interest_Curve": "",
                                                       "Funding_Benefit_Interest_Curve": "",
                                                       "Deflate_Stochastically": "Yes", "Stochastic_Funding": "No",
                                                       "Gradient": "No"},
                                            'sub_fields': ['Calculate', 'Counterparty', 'Bank', 'Risk_Free_Curve',
                                                           'Funding_Cost_Interest_Curve', 'Funding_Benefit_Interest_Curve', 
                                                           'Deflate_Stochastically', 'Stochastic_Funding',
                                                           'Gradient']},
            'Collateral_Valuation_Adjustment': {'widget': 'Container', 'description': 'Collateral Valuation Adjustment',
                                                'value': {"Calculate": "No", "Collateral_Curve": "",
                                                          "Funding_Curve": "", "Collateral_Spread": 0,
                                                          "Funding_Spread": 0, "Gradient": "No"},
                                                'sub_fields': ['Calculate', 'Collateral_Curve',
                                                               'Funding_Curve', 'Collateral_Spread',
                                                               'Funding_Spread', 'Gradient']},
            'Generate_Cashflows': {'widget': 'Dropdown', 'description': 'Generate Cashflows', 'value': 'Yes',
                                   'values': ['Yes', 'No'], 'Output': 'Cashflows'}
        },
        'types': {
            'CreditMonteCarlo': ['Base_Date', 'Currency', 'Base_Time_Grid', 'Deflation_Interest_Rate', 'Percentile',  
                                 'MCMC_Simulations', 'Simulation_Batches', 'Batch_Size', 'Random_Seed', 'Antithetic', 
                                 'Calc_Scenarios', 'Dynamic_Scenario_Dates', 'Generate_Cashflows', 'Credit_Valuation_Adjustment', 
                                 'Funding_Valuation_Adjustment', 'Collateral_Valuation_Adjustment'],
            'BaseValuation': ['Base_Date', 'Currency', 'MCMC_Simulations', 'Random_Seed', 'Greeks']
        }
    },
    'System': {
        'fields': {
            'Base_Currency': {'widget': 'Text', 'description': 'Base Currency', 'value': ''},
            'Base_Date': {'widget': 'DatePicker', 'description': 'Base Date', 'value': default['DatePicker']},
            'Exclude_Deals_With_Missing_Market_Data': {'widget': 'Dropdown',
                                                       'description': 'Exclude Deals With Missing Market Data',
                                                       'value': 'Yes', 'values': ['Yes', 'No']},
            'Correlations_Healing_Method': {'widget': 'Dropdown', 'description': 'Correlations Healing Method',
                                            'value': 'Eigenvalue_Raising',
                                            'values': ['Eigenvalue_Raising', 'Alternating_Projections']}
        },
        'types': {
            'Config':
                ['Base_Currency', 'Base_Date', 'Exclude_Deals_With_Missing_Market_Data',
                 'Correlations_Healing_Method']
        }
    },
    'Factor': {
        # All supported risk factors - need to append this once new risk factors are developed.
        'types': {
            "Correlation":
                ["Value"],
            "CommodityPrice":
                ["Spot", "Currency", "Interest_Rate"],
            "CSForwardPriceModelParameters":
                ["Sigma", "Alpha"],
            "HestonNandiModelParameters":
                ["Omega", "Alpha", "Beta", "Gamma_Star", "H0"],
            "ConvenienceYield":
                ["Curve", "Currency"],
            "FuturesPrice":
                ["Price"],
            "InterestYieldVol":
                ["Space", "Shift", "Distribution_Type"],
            "InflationRate":
                ["Price_Index", "Seasonal_Adjustment", "Reference_Name", "Day_Count", "Accrual_Calendar", "Currency",
                 "Curve"],
            "VolatilityGrid":
                _VOLATILITY_GRID,
            "EquityPrice":
                ["Issuer", "Respect_Default", "Jump_Level", "Currency", "Interest_Rate", "Spot"],
            "FxRate":
                ["Domestic_Currency", "Interest_Rate", "Priority", "Spot"],
            "SurvivalProb":
                ["Recovery_Rate", "Minimum_Recovery_Rate", "Issuer", "Curve"],
            "InterestRate":
                ["Sub_Type", "Floor", "Day_Count", "Accrual_Calendar", "Currency", "Curve", "Near_Interpolation", "Near_Date"],
            "HullWhite2FactorModelParameters":
                ["Quanto_FX_Volatility", "Alpha_1", "Sigma_1", "Quanto_FX_Correlation_1", "Alpha_2", "Sigma_2",
                 "Quanto_FX_Correlation_2", "Correlation"],
            "GBMAssetPriceTSModelParameters":
                ["Quanto_FX_Volatility", "Vol", "Quanto_FX_Correlation"],
            "PriceIndex":
                ["Index", "Next_Publication_Date", "Last_Period_Start", "Publication_Period", "Currency"],
            "ObservedBasis":
                ["Spot"],
            "ForwardPrice":
                ["Currency", "Curve", "Fixings"],
            "ForwardRate":
                ["Currency", "Curve"],
            "ForwardPriceSample":
                ["Offset", "Holiday_Calendar", "Sampling_Convention"],
            "ReferencePrice":
                ["Fixing_Curve", "ForwardPrice"],
            "ReferenceVol":
                ["ForwardPriceVol", "ReferencePrice"],
            "ForwardPriceVol":
                ["Space"],
            "InterestRateVol":
                ["Space"],
            "DividendRate":
                ["Floor", "Currency", "Curve"]
        },

        # field types for the various risk factors - need to explicitly mention all of them
        'fields': {
            'Accrual_Calendar': {'widget': 'Text', 'description': 'Accrual Calendar', 'value': ''},
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': ''},
            'Curve': {'widget': 'Flot', 'description': 'Curve', 'value': default['Flot']},
            'Fixing_Curve': {'widget': 'Flot', 'description': 'Fixing Curve', 'value': default['Flot']},
            'Day_Count': {'widget': 'Dropdown', 'description': 'Day Count', 'value': 'ACT_365',
                          'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
            'Surface_Type': {'widget': 'Dropdown', 'description': 'Surface Type', 'value': 'Explicit',
                             'values': ['Explicit', 'SVI', 'Skew', 'Malz', 'Relative_Forward']},
            'Moneyness_Rule': {'widget': 'Dropdown', 'description': 'Moneyness Rule', 'value': 'Sticky_Moneyness',
                               'values': ['Sticky_Strike', 'Sticky_Moneyness', 'Sticky_Delta']},
            'Domestic_Currency': {'widget': 'Text', 'description': 'Domestic Currency', 'value': ''},
            'Floor': {'widget': 'Text', 'description': 'Floor', 'value': '<undefined>'},
            'ForwardPrice': {'widget': 'Text', 'description': 'ForwardPrice', 'value': '', 'obj': 'Tuple'},
            'ReferencePrice': {'widget': 'Text', 'description': 'ReferencePrice', 'value': '', 'obj': 'Tuple'},
            'ForwardPriceVol': {'widget': 'Text', 'description': 'ForwardPriceVol', 'value': '', 'obj': 'Tuple'},
            'Holiday_Calendar': {'widget': 'Text', 'description': 'Holiday Calendar', 'value': ''},
            'Sampling_Convention': {'widget': 'Dropdown', 'description': 'Sampling Convention',
                                    'value': 'ForwardPriceSampleDaily',
                                    'values': ['ForwardPriceSampleDaily', 'ForwardPriceSampleBullet']},
            'Offset': {'widget': 'Integer', 'description': 'Offset', 'value': 0},
            'Value': {'widget': 'Float', 'description': 'Value', 'value': 0},
            'Sigma': {'widget': 'Float', 'description': 'Sigma', 'value': 0},
            'Shift': {'widget': 'Float', 'description': 'Shift', 'value': 0, 'obj': 'Percent'},
            'Distribution_Type': {'widget': 'Dropdown', 'description': 'Distribution Type', 'value': 'Lognormal',
                                  'values': ['Lognormal', 'Normal']},
            'Alpha': {'widget': 'Float', 'description': 'Alpha', 'value': 0},
            'Beta': {'widget': 'Float', 'description': 'Beta', 'value': 0},
            'Omega': {'widget': 'Float', 'description': 'Omega', 'value': 0},
            'Gamma_Star': {'widget': 'Float', 'description': 'Gamma Star', 'value': 0},
            'H0': {'widget': 'Float', 'description': 'H0', 'value': 0},
            'Issuer': {'widget': 'Text', 'description': 'Issuer', 'value': ''},
            'Index': {'widget': 'Flot', 'description': 'Index', 'value': default['Flot']},
            'Vol': {'widget': 'Flot', 'description': 'Vol', 'value': default['Flot']},
            'Fixings': {'widget': 'Text', 'description': 'Fixings', 'value': ''},
            'Interest_Rate': {'widget': 'Text', 'description': 'Interest Rate', 'value': '', 'obj': 'Tuple'},
            'Jump_Level': {'widget': 'Float', 'description': 'Jump Level', 'value': 0.0, 'obj': 'Percent'},
            'Last_Period_Start': {'widget': 'DatePicker', 'description': 'Last Period Start',
                                  'value': default['DatePicker']},
            'Near_Date':{'widget': 'DatePicker', 'description': 'Near Date', 'value': default['DatePicker']},
            'Correlation': {'widget': 'Float', 'description': 'Correlation', 'value': 0},
            'Quanto_FX_Correlation': {'widget': 'Float', 'description': 'Quanto FX Correlation', 'value': 0},
            'Quanto_FX_Correlation_1': {'widget': 'Float', 'description': 'Quanto FX_Correlation 1', 'value': 0},
            'Quanto_FX_Correlation_2': {'widget': 'Float', 'description': 'Quanto FX Correlation 2', 'value': 0},
            'Alpha_1': {'widget': 'Float', 'description': 'Alpha 1', 'value': 0},
            'Alpha_2': {'widget': 'Float', 'description': 'Alpha 2', 'value': 0},
            'Quanto_FX_Volatility': {'widget': 'Flot', 'description': 'Quanto FX Volatility', 'value': default['Flot']},
            'a': {'widget': 'Flot', 'description': 'a', 'value': default['Flot']},
            'b': {'widget': 'Flot', 'description': 'b', 'value': default['Flot']},
            's': {'widget': 'Flot', 'description': 's', 'value': default['Flot']},
            'L': {'widget': 'Flot', 'description': 'L', 'value': default['Flot']},
            'R': {'widget': 'Flot', 'description': 'R', 'value': default['Flot']},
            'C': {'widget': 'Flot', 'description': 'C', 'value': default['Flot']},
            'D': {'widget': 'Flot', 'description': 'D', 'value': default['Flot']},
            'lam': {'widget': 'Flot', 'description': 'lam', 'value': default['Flot']},
            'rho': {'widget': 'Flot', 'description': 'rho', 'value': default['Flot']},
            'ATM_Ref': {'widget': 'Flot', 'description': 'ATM Ref', 'value': default['Flot']},
            'ATM_Vol': {'widget': 'Flot', 'description': 'ATM Vol', 'value': default['Flot']},
            'm': {'widget': 'Flot', 'description': 'm', 'value': default['Flot']},
            'sigma': {'widget': 'Flot', 'description': 'sigma', 'value': default['Flot']},
            'Sigma_1': {'widget': 'Flot', 'description': 'Sigma 1', 'value': default['Flot']},
            'Sigma_2': {'widget': 'Flot', 'description': 'Sigma 2', 'value': default['Flot']},
            'Minimum_Recovery_Rate': {'widget': 'Text', 'description': 'Minimum Recovery Rate', 'value': '<undefined>'},
            'Next_Publication_Date': {'widget': 'DatePicker', 'description': 'Next Publication Date',
                                      'value': default['DatePicker']},
            'Price_Index': {'widget': 'Text', 'description': 'Price Index', 'value': '', 'obj': 'Tuple'},
            'Priority': {'widget': 'Float', 'description': 'Priority', 'value': 3},
            'Publication_Period': {'widget': 'Dropdown', 'description': 'Publication Period', 'value': 'Monthly',
                                   'values': ['Monthly', 'Quarterly']},
            'Near_Interpolation':{'widget': 'Text', 'description': 'Near Interpolation', 'value': ''},
            'Reference_Name': {'widget': 'Dropdown', 'description': 'Reference Name',
                               'value': 'IndexReferenceInterpolated3M',
                               'values': ['IndexReferenceInterpolated1M', 'IndexReferenceInterpolated2M',
                                          'IndexReferenceInterpolated3M', 'IndexReferenceInterpolated4M']},
            'Respect_Default': {'widget': 'Dropdown', 'description': 'Respect Default', 'value': 'Yes',
                                'values': ['Yes', 'No']},
            'Recovery_Rate': {'widget': 'BoundedFloat', 'description': 'Recovery Rate', 'value': 0.4, 'min': 0.0,
                              'max': 1.0},
            'Seasonal_Adjustment': {'widget': 'Text', 'description': 'Seasonal Adjustment', 'value': ''},
            'Spot': {'widget': 'Float', 'description': 'Spot', 'value': 0},
            'Price': {'widget': 'Float', 'description': 'Price', 'value': 0},
            'Surface': {'widget': 'Three', 'description': 'Surface', 'value': default['Surface']},
            'Delta_Surface': {'widget': 'Three', 'description': 'Delta_Surface', 'value': default['Surface']},
            'Space': {'name': 'Surface', 'widget': 'Three', 'description': 'Surface', 'value': default['Space']},
            'Sub_Type': {'widget': 'Text', 'description': 'Sub Type', 'value': ''}
        }
    },
    'Process': {
        # All supported risk stochastic processes - need to append this once new risk processes are developed.
        'types': {
            "GBMAssetPriceTSModelImplied":
                ["Risk_Premium"],
            "HestonNandiImpliedSpotModel":
                [],
            "HullWhite2FactorImpliedInterestRateModel":
                ["Lambda_1", "Lambda_2"],
            "GBMAssetPriceModel":
                ["Vol", "Drift"],
            "GBMPriceIndexModel":
                ["Vol", "Drift", "Seasonal_Adjustment"],
            "HWHazardRateModel":
                ["Alpha", "Lambda", "sigma"],
            "LogOUSpotModel":
                ["Kappa", "Theta", "sigma"],
            "MarkovSwitchingLogOUSpotModel":
                ["States", "Transition_Matrix", "Initial_State_Probs", "Calibration_DT_Years"],
            "MarkovHMMSpotModel":
                ["States", "Transition_Matrix", "Initial_State_Probs", "Calibration_DT_Years"],
            "VARMixedFactorInterestRateModel":
                ["Mean", "Phi", "Sigma", "Calibration_Tenors", "Contract_Cycle_Years",
                 "Calibration_DT_Years"],
            "BasisLinkedSpotModel":
                ["A", "Phi", "Nu", "Sigma_By_State", "Mu", "Calibration_DT_Years"],
            "SingleRegimeOU1FactorKalmanModel":
                ["Kappa", "Theta", "sigma"],
            "PCAInterestRateModel":
                ["Reversion_Speed", "Historical_Yield", "Yield_Volatility", "Eigenvectors", "Rate_Drift_Model",
                 "Princ_Comp_Source", "Distribution_Type"],
            "CSForwardPriceModel":
                ["Alpha", "Drift", "sigma"],
            "CSImpliedForwardPriceModel":
                [],
            "HullWhite1FactorInterestRateModel":
                ["Alpha", "Lambda", "Sigma", "Quanto_FX_Correlation", "Quanto_FX_Volatility"]
        },

        # field types for the various risk processes - need to explicitly mention all of them
        'fields': {
            'Vol': {'widget': 'Float', 'description': 'Vol', 'value': 0},
            'Drift': {'widget': 'Float', 'description': 'Drift', 'value': 0},
            'Alpha': {'widget': 'Float', 'description': 'Alpha', 'value': 0},
            'Kappa': {'widget': 'Float', 'description': 'Kappa', 'value': 0},
            'Theta': {'widget': 'Float', 'description': 'Theta', 'value': 0},
            'Lambda': {'widget': 'Float', 'description': 'Lambda', 'value': 0},
            'Lambda_1': {'widget': 'Float', 'description': 'Lambda 1', 'value': 0},
            'Lambda_2': {'widget': 'Float', 'description': 'Lambda 2', 'value': 0},
            'sigma': {'name': 'Sigma', 'widget': 'Float', 'description': 'Sigma', 'value': 0},
            'Risk_Premium': {'widget': 'Flot', 'description': 'Risk Premium', 'value': default['Flot']},
            'Quanto_FX_Correlation': {'widget': 'Float', 'description': 'Quanto_FX_Correlation', 'value': 0},
            'Reversion_Speed': {'widget': 'Float', 'description': 'Reversion Speed', 'value': 0},
            'Historical_Yield': {'widget': 'Flot', 'description': 'Historical Yield', 'value': default['Flot']},
            'Yield_Volatility': {'widget': 'Flot', 'description': 'Yield Volatility', 'value': default['Flot']},
            'Sigma': {'widget': 'Flot', 'description': 'Sigma', 'value': default['Flot']},
            'Rate_Drift_Model': {'widget': 'Dropdown', 'description': 'Rate Drift Model', 'value': 'Drift_To_Forward',
                                 'values': ['Drift_To_Forward', 'Drift_To_Blend']},
            'Princ_Comp_Source': {'widget': 'Dropdown', 'description': 'Princ Comp Source', 'value': 'Correlation',
                                  'values': ['Correlation', 'Covariance']},
            'Distribution_Type': {'widget': 'Dropdown', 'description': 'Distribution Type', 'value': 'Lognormal',
                                  'values': ['Lognormal', 'Normal']},
            'Eigenvectors': {'widget': 'Flot', 'description': 'Eigenvectors',
                             'value': '[{"label":"1", "data":[[0.0,0.0]]},{"label":"2", "data":[[0.0,0.0]]},{"label":"3", "data":[[0.0,0.0]]}]'},
            'Quanto_FX_Volatility': {'widget': 'Flot', 'description': 'Quanto FX Volatility',
                                     'value': default['Flot']},
            'Seasonal_Adjustment': {'widget': 'Text', 'description': 'Seasonal Adjustment', 'value': ''},
            # MarkovSwitchingLogOUSpotModel fields. The latent regime z_t follows a Markov chain
            # with transition matrix P; conditional on z_t the log-spot follows a per-regime OU.
            'States': {'widget': 'Container', 'description':
                       'List of per-regime {Kappa, Theta, Sigma} dicts (must have at least 2 regimes)',
                       'value': []},
            'Transition_Matrix': {'widget': 'Table', 'description':
                                  'NxN row-stochastic transition matrix at the calibration time step',
                                  'value': []},
            'Initial_State_Probs': {'widget': 'Table', 'description':
                                    'Initial regime distribution (length-N vector summing to 1)',
                                    'value': []},
            'Calibration_DT_Years': {'widget': 'Float', 'description':
                                     'Step size (in years) of the calibrated transition matrix; the model '
                                     're-discretises P per simulation step via the CTMC generator',
                                     'value': 1.0 / 252.0},
            # MarkovHMMSpotModel: per-state {Mu, Sigma} are annualised drift/vol of additive ΔS.
            'Mu': {'widget': 'Float', 'description': 'Annualised additive drift (per-state)', 'value': 0.0},
            # VARMixedFactorInterestRateModel: 3-factor VAR(1) on (β_0, β_1, r) with curvature
            # weight w from the orthogonal-to-(1,1,1)-and-τ construction.
            'Mean': {'widget': 'Container', 'description':
                     'Long-run mean vector [μ_β0, μ_β1, μ_r]',
                     'value': []},
            'Phi': {'widget': 'Table', 'description':
                    'VAR(1) transition matrix Φ (3x3) at the calibration step',
                    'value': []},
            'Calibration_Tenors': {'widget': 'Container', 'description':
                                   'Slot tenor vector τ_i(0) at simulation start (years)',
                                   'value': []},
            'Contract_Cycle_Years': {'widget': 'Float', 'description':
                                     'Front-slot roll cycle (years) — contracts shift forward by this '
                                     'amount once the front slot expires', 'value': 0.25},
            # BasisLinkedSpotModel: lagged-AR(1) basis on a sibling commodity-spot path.
            'A': {'widget': 'Float', 'description': 'Concurrent ΔS loading on the basis', 'value': 0.0},
            'Nu': {'widget': 'Float', 'description': 'Student-t degrees of freedom (basis innovation)', 'value': 5.0},
            'Sigma_By_State': {'widget': 'Container', 'description':
                               'Per-regime innovation std σ_s (indexed by linked-spot HMM state)',
                               'value': []},
        },
    },

    # list mapping risk factors to allowable interpolation methods
    'Interpolation_factor_map': {
        "InflationRate": ['HermiteRT','Hermite','LinearRT','Linear'],
        "InterestRate":['HermiteRT','Hermite','LinearRT','Linear']
    },
    # list mapping risk factors to allowable stochastic processes
    'Process_factor_map': {
        "Correlation": [],
        "CommodityPrice": ['LogOUSpotModel', 'MarkovSwitchingLogOUSpotModel', 'MarkovHMMSpotModel'],
        "ConvenienceYield": [],
        "ObservedBasis": ["SingleRegimeOU1FactorKalmanModel", "BasisLinkedSpotModel"],
        "InterestYieldVol": [],
        "FuturesPrice": [],
        "InflationRate": ["HullWhite1FactorInterestRateModel", "PCAInterestRateModel"],
        "VolatilityGrid": [],
        "ForwardPrice": ["CSForwardPriceModel"],
        "ForwardRate": ["VARMixedFactorInterestRateModel"],
        "ForwardPriceVol": [],
        "ForwardPriceSample": [],
        "ReferencePrice": [],
        "ReferenceVol": [],
        "HullWhite2FactorModelParameters": [],
        # "GBMTSImpliedParameters": [],
        "CSForwardPriceModelParameters": [],
        "HestonNandiModelParameters": [],
        "GBMAssetPriceTSModelParameters": [],
        "EquityPrice": ["GBMAssetPriceModel", "HestonNandiImpliedSpotModel"],
        "FxRate": ["GBMAssetPriceModel", "GBMAssetPriceTSModelImplied", "HestonNandiImpliedSpotModel"],
        "SurvivalProb": ["HWHazardRateModel"],
        "InterestRate": ["HullWhite1FactorInterestRateModel", "PCAInterestRateModel",
                          "VARMixedFactorInterestRateModel"],
        "PriceIndex": ["GBMPriceIndexModel"],
        "InterestRateVol": [],
        "DividendRate": ["HullWhite1FactorInterestRateModel", "PCAInterestRateModel"]
    },
    'MarketPrices': {
        # logical groupings
        'groups': {
            'MarketPrices': (
                'group', ['InterestRatePrices', 'GBMAssetPriceTSModelPrices', 'HullWhite2FactorModelPrices',
                          'HestonNandiModelPrices', 'CSForwardPriceModelPrices']),
            'PointFields': ('default', ['FRADeal', 'SwapInterestDeal', 'DepositDeal']),
        },

        # field groups
        'sections': {
            'InterestRatePrices':
                ['FRADeal', 'SwapInterestDeal', 'DepositDeal'],
            'GBMAssetPriceTSModelPrices':
                [],
            'HullWhite2FactorModelPrices':
                [],
            'HestonNandiModelPrices':
                [],
            'CSForwardPriceModelPrices':
                []
        },

        # supported types
        'types': {
            "InterestRatePrices":
                ["Currency", "Spot_Offset", "Zero_Rate_Grid", "Discount_Rate"],
            "GBMAssetPriceTSModelPrices":
                ["Asset_Price_Volatility"],
            "HullWhite2FactorModelPrices":
                ["Swaption_Volatility", "Generate_Instruments", "Generation_Parameters", "Instrument_Definitions"],
            "CSForwardPriceModelPrices":
                ["Energy", "Forward_Volatility", "Discount_Rate", "Quote_Type", "Energy_Futures_Options"],
            "HestonNandiModelPrices":
                ["Underlying", "Underlying_Type", "Volatility", "Volatility_Type", "Discount_Rate",
                 "Yield", "Yield_Type", "Quote_Type", "Use_Forward", "Invert_Moneyness",
                 "Steps_Per_Year", "Quadrature_Panels", "European_Options"],
            "quote":
                ["Descriptor", "Use", "Quoted_Market_Value", "DealType", "Quote_Type"]
        },

        'properties': {
            'Locked_Dates': ['Maturity_Date', 'Effective_Date'],
        },

        # instrument fields
        'fields': {
            'Generation_Parameters': {'widget': 'Container', 'description': 'Generation Parameters',
                                      'value': {"Last_Tenor": "9Y", "Floating_Frequency": "6M", "First_Tenor": "1Y",
                                                "Day_Count": "ACT_365", "Last_Maturity": "10Y", "First_Start": "1Y",
                                                "Fixed_Frequency": "6M", "Index_Offset": 0, "Last_Start": "9Y",
                                                "First_Maturity": "10Y"},
                                      'sub_fields': ["Last_Tenor", "Floating_Frequency", "First_Tenor", "Day_Count",
                                                     "Last_Maturity", "First_Start", "Fixed_Frequency", "Index_Offset",
                                                     "Last_Start", "First_Maturity"]},
            'Swaption_Volatility': {'widget': 'Text', 'description': 'Swaption Volatility', 'value': ''},
            'Fixed_Frequency': {'widget': 'Text', 'description': 'Fixed Frequency', 'value': '6M', 'obj': 'Period'},
            'Floating_Frequency': {'widget': 'Text', 'description': 'Floating Frequency', 'value': '6M',
                                   'obj': 'Period'},
            'First_Start': {'widget': 'Text', 'description': 'First Start', 'value': '1Y', 'obj': 'Period'},
            'Last_Start': {'widget': 'Text', 'description': 'Last Start', 'value': '9Y', 'obj': 'Period'},
            'First_Tenor': {'widget': 'Text', 'description': 'First Tenor', 'value': '1Y', 'obj': 'Period'},
            'Last_Tenor': {'widget': 'Text', 'description': 'Last Tenor', 'value': '9Y', 'obj': 'Period'},
            'First_Maturity': {'widget': 'Text', 'description': 'First Maturity', 'value': '10Y', 'obj': 'Period'},
            'Last_Maturity': {'widget': 'Text', 'description': 'Last Maturity', 'value': '10Y', 'obj': 'Period'},
            'Day_Count': {'widget': 'Dropdown', 'description': 'Day Count', 'value': 'ACT_365',
                          'values': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360', 'ACT_ACT_ICMA']},
            'Generate_Instruments': {'widget': 'Dropdown', 'description': 'Generate Instruments', 'value': 'No',
                                     'values': ['Yes', 'No']},
            'Index_Offset': {'widget': 'Integer', 'description': 'Index Offset', 'value': 0},
            'Holiday_Calendar': {'widget': 'Text', 'description': 'Holiday Calendar', 'value': ''},
            'Instrument_Definitions': {'widget': 'Table', 'description': 'Instrument Definitions', 'value': 'null',
                                       'sub_types':
                                           [{},
                                            {'type': 'numeric', 'numericFormat': num_format['currency']},
                                            {},
                                            {'type': 'dropdown',
                                             'source': ['ACT_365', 'ACT_360', 'ACT_365_ISDA', '_30_360', '_30E_360',
                                                        'ACT_ACT_ICMA']},
                                            {},
                                            {},
                                            {},
                                            {'type': 'dropdown', 'source': ['Lognormal', 'Normal']},
                                            {'type': 'numeric', 'numericFormat': num_format['int']},
                                            {'type': 'numeric', 'numericFormat': num_format['percent']}],
                                       'obj':
                                           ['Period', 'Period', 'Period', 'Period', 'Text', 'Integer', 'Text', 'Float',
                                            'Percent', 'Text'],
                                       'col_names':
                                           ['Floating_Frequency', 'Weight', 'Holiday_Calendar', 'Day_Count', 'Start',
                                            'Fixed_Frequency', 'Tenor', 'Market_Volatility_Type', 'Index_Offset',
                                            'Market_Volatility']
                                       },
            'Descriptor': {'widget': 'Text', 'description': 'Descriptor', 'value': ''},
            'Discount_Rate': {'widget': 'Text', 'description': 'Discount Rate', 'value': ''},
            'Currency': {'widget': 'Text', 'description': 'Currency', 'value': ''},
            'Asset_Price_Volatility': {'widget': 'Text', 'description': 'Asset Price Volatility', 'value': ''},
            'Energy': {'widget': 'Text', 'description': 'Energy forward price factor', 'value': ''},
            'Forward_Volatility': {'widget': 'Text', 'description': 'Forward Volatility', 'value': ''},
            'Energy_Futures_Options': {'widget': 'Table', 'description': 'Energy Futures Options', 'value': 'null',
                                       'col_names': ['Expiry_Date', 'Settlement_Date', 'Strike', 'Option_Type',
                                                     'Units', 'Quoted_Market_Value'],
                                       'obj': ['DatePicker', 'DatePicker', 'Float', 'Text', 'Float', 'Float']},
            'Spot_Offset': {'widget': 'Integer', 'description': 'Spot Offset', 'value': 2},
            'Zero_Rate_Grid': {'widget': 'Text', 'description': 'Zero Rate Grid',
                               'value': '0d 1d 2d 1w 2w 1m 3m 6m 9m 1y 6m1y 2y 6m2y 3y 6m3y 4y 6m4y 5y 6y 7y 8y 9y 10y 15y 20y 25y'},
            'Points': {'widget': 'Container', 'description': 'Points',
                       'value': {"Use": "Yes", "Deal": "", "Descriptor": "", "Quote_Type": "ATM", "DealType": "",
                                 "Quoted_Market_Value": 0.0},
                       'sub_fields': ['Use', 'Deal', 'Descriptor', 'Quote_Type', 'DealType', 'Quoted_Market_Value']},
            'Quote_Type': {'widget': 'Dropdown', 'description': 'Quote Type', 'value': 'ATM',
                           'values': ['ATM', 'Implied_Volatility', 'Premium']},
            # the Heston-Nandi inputs are asset class agnostic - the *_Type fields are optional and
            # only needed to disambiguate a name that exists under more than one factor type
            'Underlying': {'widget': 'Text', 'description': 'Underlying spot price factor', 'value': ''},
            'Underlying_Type': {'widget': 'Dropdown', 'description': 'Underlying Type', 'value': '',
                                'values': ['', 'FxRate', 'EquityPrice', 'CommodityPrice', 'FuturesPrice']},
            'Volatility': {'widget': 'Text', 'description': 'Volatility surface price factor', 'value': ''},
            'Volatility_Type': {'widget': 'Dropdown', 'description': 'Volatility Type', 'value': '',
                                'values': ['', 'VolatilityGrid']},
            'Yield': {'widget': 'Text', 'description': 'Dividend/repo/convenience yield curve', 'value': ''},
            'Yield_Type': {'widget': 'Dropdown', 'description': 'Yield Type', 'value': '',
                           'values': ['', 'DividendRate', 'InterestRate']},
            # the two moneyness flags pricing.calc_moneyness takes - defaults match the pricing path
            'Use_Forward': {'widget': 'Dropdown', 'description': 'Use Forward', 'value': 'No',
                            'values': ['Yes', 'No']},
            'Invert_Moneyness': {'widget': 'Dropdown', 'description': 'Invert Moneyness', 'value': 'No',
                                 'values': ['Yes', 'No']},
            'Steps_Per_Year': {'widget': 'Float', 'description': 'Steps Per Year', 'value': 252.0},
            'Quadrature_Panels': {'widget': 'Integer', 'description': 'Quadrature Panels', 'value': 64},
            'European_Options': {'widget': 'Table', 'description': 'European Options', 'value': 'null',
                                 'col_names': ['Expiry_Date', 'Strike', 'Option_Type', 'Units', 'Weight',
                                               'Quoted_Market_Value'],
                                 'obj': ['DatePicker', 'Float', 'Text', 'Float', 'Float', 'Float']},
            'Use': {'widget': 'Dropdown', 'description': 'Use', 'value': 'Yes', 'values': ['Yes', 'No']},
            'DealType': {'widget': 'Dropdown', 'description': 'DealType', 'value': 'DepositDeal',
                         'values': ['DepositDeal', 'FRADeal', 'SwapInterestDeal']},
            'Quoted_Market_Value': {'widget': 'Float', 'description': 'Quoted Market Value', 'value': 0.0}
        }
    },
    'Instrument': {
        # logical groupings
        'groups': {
            'New Structure': ('group', ['NettingCollateralSet', 'StructuredDeal']),
            'New Interest Rate Derivative': (
                'default', ['FixedCashflowDeal', 'CFFixedListDeal', 'CFFixedInterestListDeal',
                            'CFFloatingInterestListDeal', 'DepositDeal', 'CapDeal', 'FRADeal',
                            'FloorDeal', 'SwapInterestDeal', 'SwaptionDeal',
                            'YieldInflationCashflowListDeal', 'CashAccountDeal']),
            'New FX Derivative': (
                'default', ['FXNonDeliverableForward', 'FXForwardDeal', 'FXOptionDeal', 'FXBinaryOption',
                            'FXDiscreteExplicitAsianOption', 'FXOneTouchOption',
                            'FXBarrierOption', 'FXSwapDeal',
                            'MtMCrossCurrencySwapDeal', 'FXTARFOptionDeal',
                            'FXDiscreteExplicitDoubleAsianOption', 'FXPartialTimeBarrierOption']),
            'New Energy Derivative': (
                'default', ['FloatingEnergyDeal', 'FixedEnergyDeal', 'EnergySingleOption', 'CommodityForwardDeal',
                            'CommodityFutureDeal']),
            'New Equity Derivative': ('default', ['EquityDeal', 'EquitySwapLeg', 'EquityForwardDeal',
                                                  'EquityOptionDeal', 'EquityBinaryOption',
                                                  'EquityOneTouchOption', 'QEDI_CustomAutoCallSwap',
                                                  'QEDI_CustomAutoCallSwap_V2', 'EquitySwapletListDeal',
                                                  'EquityBarrierOption', 'EquityBarrierBinaryOption',
                                                  'EquityDiscreteExplicitAsianOption']),
            'New Credit Derivative': ('default', ['DealDefaultSwap','CreditNthToDefault'])
        },

        'sections': _sections,
        'types': _types
    }
}
