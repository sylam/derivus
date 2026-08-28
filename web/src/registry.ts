// The workspace registry: every view is an entry, and a future one - a SACCR screen, a backtest,
// a market-data archive browser - is a NEW ENTRY, not a refactor. The house pattern (registries,
// not functions) moved into the client.

import type { ComponentType } from 'react';
import { BlotterView } from './views/BlotterView';
import { CalculationView } from './views/CalculationView';
import { MarketDataView } from './views/MarketDataView';
import { PortfolioView } from './views/PortfolioView';
import { RiskView } from './views/RiskView';
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
  { id: 'market', label: 'Market Data', view: MarketDataView },
  { id: 'calculation', label: 'Calculation', view: CalculationView },
  { id: 'settings', label: 'Settings', view: SettingsView },
];
