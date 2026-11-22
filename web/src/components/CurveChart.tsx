import { useMemo } from 'react';
import { chartPalette, useChart } from '../charts/useChart';
import { axisText, formatNumber } from '../tokens';
import { axisNames } from '../vols';

/** A 1-D `.Curve` - `[[x, y], ...]` - as a line with points, the axes quiet, the data loud. Its
 * `description` names the two coordinates, which is how x reads: a tenor or a date. */
export function CurveChart({ data, description }: { data: number[][]; description?: string }) {
  const option = useMemo(() => {
    const palette = chartPalette();
    const [x] = axisNames(description, 2);
    return {
      animation: false,
      grid: { left: 60, right: 40, top: 20, bottom: 36 },
      tooltip: {
        trigger: 'axis' as const,
        formatter: (points: { value: number[] }[]) =>
          `${axisText(x, points[0].value[0])}<br/>${formatNumber(points[0].value[1])}`,
      },
      xAxis: {
        type: 'value' as const, scale: true,
        axisLine: { lineStyle: { color: palette.axis } },
        axisLabel: { color: palette.text, formatter: (value: number) => axisText(x, value) },
        splitLine: { lineStyle: { color: palette.axis, opacity: 0.5 } },
      },
      yAxis: {
        type: 'value' as const, scale: true,
        axisLabel: { color: palette.text },
        splitLine: { lineStyle: { color: palette.axis, opacity: 0.5 } },
      },
      series: [{
        name: 'value', type: 'line' as const,
        symbolSize: 5, data,
        lineStyle: { color: palette.accent, width: 2 },
        itemStyle: { color: palette.accent },
      }],
    };
  }, [data, description]);

  return <div className="chart" ref={useChart(option)} />;
}
