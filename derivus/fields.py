from . import bootstrappers, calculation, instruments, riskfactors, stochasticprocess
from .schema import (BLANK, emit_calculation, emit_calibration, emit_factor, emit_instrument,
                     emit_interpolation, emit_market_prices, emit_process)

# the Instrument, Factor, Process, Calculation and Calibration stores are VIEWS of the per-class
# declarations, not a second copy
_types, _sections = emit_instrument(instruments)
_factor_types = emit_factor(riskfactors)
_process_types, _process_factor_map = emit_process(stochasticprocess, _factor_types)
_calculation_types = emit_calculation(calculation)
_calibration_types = emit_calibration(stochasticprocess)
_interpolation_factor_map = emit_interpolation(riskfactors)
_market_price_types = emit_market_prices(bootstrappers)

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
    # the Calibration store is a VIEW, keyed by the PROCESS a `Calibrations` entry is filed under
    # while its `Method` names the calibration class the engine dispatches on
    'Calibration': {'types': _calibration_types},

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

    # the UI's interpolation-per-factor menu, and a VIEW too: a curve factor declares the methods
    # it can be set to, and only the types `construct_factor` routes through this section have any
    'Interpolation_factor_map': _interpolation_factor_map,
    # the UI's valid-processes-per-factor menu, the process declarations read the other way
    # round. Every factor type is a key, including the ones no process drives.
    'Process_factor_map': _process_factor_map,
    # the MarketPrices store is a VIEW: a price FAMILY holds its own descriptors, keyed by the
    # `Market Prices` type string the engine selects work by
    'MarketPrices': {'types': _market_price_types},
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
