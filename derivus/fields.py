from . import calculation, instruments, riskfactors, stochasticprocess
from .schema import BLANK, emit_calculation, emit_factor, emit_instrument, emit_process

# the Instrument, Factor, Process and Calculation stores are VIEWS of the per-class declarations,
# not a second copy
_types, _sections = emit_instrument(instruments)
_factor_types = emit_factor(riskfactors)
_process_types, _process_factor_map = emit_process(stochasticprocess, _factor_types)
_calculation_types = emit_calculation(calculation)

# object list defaults, keyed by WIDGET - the shape-valued ones come from the declaration
# vocabulary so a blank curve has one definition
default = {
    'Integer': 0,
    'Float': 0.0,
    'Percent': 0.0,
    'Text': '',
    'Flot': BLANK['Curve'],
    'Surface': BLANK['Surface'],
    'Space': BLANK['Space'],
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

    # the Calculation store is a VIEW too, keyed by the `Object` string a job document writes
    'Calculation': {'types': _calculation_types},
    # `System` stays hand-written. Its one "type" is a UI panel name rather than anything the JSON
    # dispatches on, and the class that consumes `System Parameters` is `Config` itself - the whole
    # configuration object, so giving it a `fields` list would make "a class that declares fields
    # IS a type" mean something else in that module.
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
    # the Factor store is a VIEW: a type IS the descriptors its class declares
    'Factor': {'types': _factor_types},
    # the Process store is a VIEW too: a process TYPE holds its own descriptors
    'Process': {'types': _process_types},

    # list mapping risk factors to allowable interpolation methods
    'Interpolation_factor_map': {
        "InflationRate": ['HermiteRT','Hermite','LinearRT','Linear'],
        "InterestRate":['HermiteRT','Hermite','LinearRT','Linear']
    },
    # the UI's valid-processes-per-factor menu, the process declarations read the other way
    # round. Every factor type is a key, including the ones no process drives.
    'Process_factor_map': _process_factor_map,
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
        # the create-deal menu. Whether a type can hold children is NOT here: it is
        # `Deal.accepts_children`, because it is a property of the deal, not of the menu.
        'groups': {
            'New Structure': ['NettingCollateralSet', 'StructuredDeal'],
            'New Interest Rate Derivative':
                ['FixedCashflowDeal', 'CFFixedListDeal', 'CFFixedInterestListDeal',
                 'CFFloatingInterestListDeal', 'DepositDeal', 'CapDeal', 'FRADeal',
                 'FloorDeal', 'SwapInterestDeal', 'SwaptionDeal',
                 'YieldInflationCashflowListDeal', 'CashAccountDeal'],
            'New FX Derivative':
                ['FXNonDeliverableForward', 'FXForwardDeal', 'FXOptionDeal', 'FXBinaryOption',
                 'FXDiscreteExplicitAsianOption', 'FXOneTouchOption',
                 'FXBarrierOption', 'FXSwapDeal',
                 'MtMCrossCurrencySwapDeal', 'FXTARFOptionDeal',
                 'FXDiscreteExplicitDoubleAsianOption', 'FXPartialTimeBarrierOption'],
            'New Energy Derivative':
                ['FloatingEnergyDeal', 'FixedEnergyDeal', 'EnergySingleOption', 'CommodityForwardDeal',
                 'CommodityFutureDeal'],
            'New Equity Derivative':
                ['EquityDeal', 'EquitySwapLeg', 'EquityForwardDeal',
                 'EquityOptionDeal', 'EquityBinaryOption',
                 'EquityOneTouchOption', 'QEDI_CustomAutoCallSwap',
                 'QEDI_CustomAutoCallSwap_V2', 'EquitySwapletListDeal',
                 'EquityBarrierOption', 'EquityBarrierBinaryOption',
                 'EquityDiscreteExplicitAsianOption'],
            'New Credit Derivative': ['DealDefaultSwap', 'CreditNthToDefault']
        },

        'sections': _sections,
        'types': _types
    }
}
