# Energy

One family: `CSForwardPriceModelParameters` fits the one-factor
[Clewlow-Strickland model](../theory/energy_pricing.md) to options on energy futures, quoted as
Black vols in a [`CSForwardPriceModelPrices`](../json/market_prices.md#csforwardpricemodelprices)
block against a forward curve, its vol surface and a discount curve. It writes the
`CSForwardPriceModelParameters` factor `CSImpliedForwardPriceModel` simulates under, and a block
whose factor is already written starts from it rather than from `Seed`. All seven families are
summarised on [General](general.md).
