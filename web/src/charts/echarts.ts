// The one echarts registration, tree-shaken: lines, heatmaps and the components they need. The
// 3-D surface registers `echarts-gl` onto this same core from its own lazily imported module, and
// `useChart` imports this one the first time a chart is drawn, never on the first paint.

import * as echarts from 'echarts/core';
// a heatmap on VALUE axes is a custom series: echarts' own needs two category axes
import { CustomChart, LineChart } from 'echarts/charts';
import {
  DataZoomComponent, GridComponent, LegendComponent, TitleComponent, TooltipComponent,
  VisualMapComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  LineChart, CustomChart, GridComponent, TooltipComponent, LegendComponent,
  DataZoomComponent, VisualMapComponent, TitleComponent, CanvasRenderer,
]);

export default echarts;
