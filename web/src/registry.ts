// The workspace registry: every view is an entry, and a future one - a SACCR screen, a backtest,
// a market-data archive browser - is a NEW ENTRY, not a refactor. The house pattern (registries,
// not functions) moved into the client.

import type { ComponentType } from 'react';
import { BlotterView } from './views/BlotterView';
import { BootstrapperView } from './views/BootstrapperView';
import { CalculationView } from './views/CalculationView';
import { CurvesView } from './views/CurvesView';
import { MarketDataView } from './views/MarketDataView';
import { MarketPricesView } from './views/MarketPricesView';
import { PortfolioView } from './views/PortfolioView';
import { RiskView } from './views/RiskView';
import { SecuritiesView } from './views/SecuritiesView';
import { SettingsView } from './views/SettingsView';
import { XvaView } from './views/XvaView';

export type Workspace = { id: string; label: string; view: ComponentType };

export const WORKSPACES: Workspace[] = [
  { id: 'portfolio', label: 'Portfolio', view: PortfolioView },
  { id: 'blotter', label: 'Blotter', view: BlotterView },
  // the desk's two data views, beside the blotter they belong to: what the book is worth and what
  // it moves with, then what it costs per counterparty
  { id: 'risk', label: 'Risk', view: RiskView },
  { id: 'xva', label: 'XVA', view: XvaView },
  // the market, in the order it is built: the quotes, the benchmark rows a curve is set up from,
  // the dials that turn them into factors, the factors themselves
  { id: 'prices', label: 'Market Prices', view: MarketPricesView },
  { id: 'curves', label: 'Curves', view: CurvesView },
  { id: 'bootstrap', label: 'Bootstrapper', view: BootstrapperView },
  // the tickers underneath all of it: what this desk could quote, what a terminal verified about
  // those claims, and which print every knot was solved from
  { id: 'securities', label: 'Securities', view: SecuritiesView },
  { id: 'market', label: 'Market Data', view: MarketDataView },
  { id: 'calculation', label: 'Calculation', view: CalculationView },
  { id: 'settings', label: 'Settings', view: SettingsView },
];
