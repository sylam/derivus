import { useMemo } from 'react';
import { chartPalette } from '../charts/echarts';
import { useChart } from '../charts/useChart';

/** A 1-D `.Curve` - `[[x, y], ...]` - as a line with points, the axes quiet, the data loud. */
export function CurveChart({ data, name }: { data: number[][]; name?: string }) {
  const option = useMemo(() => {
    const palette = chartPalette();
    return {
      animation: false,
      grid: { left: 60, right: 20, top: 20, bottom: 36 },
      tooltip: { trigger: 'axis' as const, valueFormatter: (v: unknown) => String(v) },
      xAxis: {
        type: 'value' as const,
        axisLine: { lineStyle: { color: palette.axis } },
        axisLabel: { color: palette.text },
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
  }, [data, name]);

  return <div className="chart" ref={useChart(option)} />;
}
