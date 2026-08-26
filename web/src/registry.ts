// The workspace registry: every view is an entry, and a future one - a SACCR screen, a backtest,
// a market-data archive browser - is a NEW ENTRY, not a refactor. The house pattern (registries,
// not functions) moved into the client.

import type { ComponentType } from 'react';
import { CalculationView } from './views/CalculationView';
import { MarketDataView } from './views/MarketDataView';
import { PortfolioView } from './views/PortfolioView';
import { SettingsView } from './views/SettingsView';

export type Workspace = { id: string; label: string; view: ComponentType };

export const WORKSPACES: Workspace[] = [
  { id: 'portfolio', label: 'Portfolio', view: PortfolioView },
  { id: 'market', label: 'Market Data', view: MarketDataView },
  { id: 'calculation', label: 'Calculation', view: CalculationView },
  { id: 'settings', label: 'Settings', view: SettingsView },
];
