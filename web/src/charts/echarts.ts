// The one echarts registration, tree-shaken: lines, heatmaps and the components they need. A
// later 3-D surface is a lazy `import('echarts-gl')` in its own component, never here.

import * as echarts from 'echarts/core';
import { HeatmapChart, LineChart } from 'echarts/charts';
import {
  DataZoomComponent, GridComponent, LegendComponent, TitleComponent, TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  LineChart, HeatmapChart, GridComponent, TooltipComponent, LegendComponent,
  DataZoomComponent, VisualMapComponent, TitleComponent, CanvasRenderer,
]);

export default echarts;

/** The chart text/axis palette follows the app's own tokens, read off the live stylesheet. */
export function chartPalette() {
  const style = getComputedStyle(document.documentElement);
  return {
    text: style.getPropertyValue('--text-muted').trim() || '#5b6477',
    axis: style.getPropertyValue('--border').trim() || '#e3e7ee',
    accent: style.getPropertyValue('--accent').trim() || '#2563eb',
  };
}
