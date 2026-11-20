import { useMemo } from 'react';
import { chartPalette, useChart } from '../charts/useChart';
import { axisText, formatNumber } from '../tokens';
import { axisNames } from '../vols';

/** A 1-D `.Curve` - `[[x, y], ...]` - as a line with points, the axes quiet, the data loud. `axes`
 * names the two coordinates as the declaration does, which is how x reads: a tenor or a date. */
export function CurveChart({ data, name, axes = axisNames(undefined, 2) }: {
  data: number[][]; name?: string; axes?: string[];
}) {
  const option = useMemo(() => {
    const palette = chartPalette();
    return {
      animation: false,
      grid: { left: 60, right: 40, top: 20, bottom: 36 },
      tooltip: {
        trigger: 'axis' as const,
        formatter: (points: { value: number[] }[]) =>
          `${axisText(axes[0], points[0].value[0])}<br/>${formatNumber(points[0].value[1])}`,
      },
      xAxis: {
        type: 'value' as const, scale: true,
        axisLine: { lineStyle: { color: palette.axis } },
        axisLabel: { color: palette.text, formatter: (value: number) => axisText(axes[0], value) },
        splitLine: { lineStyle: { color: palette.axis, opacity: 0.5 } },
      },
      yAxis: {
        type: 'value' as const, scale: true,
        axisLabel: { color: palette.text },
        splitLine: { lineStyle: { color: palette.axis, opacity: 0.5 } },
      },
      series: [{
        name: name ?? 'value', type: 'line' as const,
        symbolSize: 5, data,
        lineStyle: { color: palette.accent, width: 2 },
        itemStyle: { color: palette.accent },
      }],
    };
  }, [data, name, axes]);

  return <div className="chart" ref={useChart(option)} />;
}
