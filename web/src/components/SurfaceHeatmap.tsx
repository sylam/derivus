import { useMemo, useState } from 'react';
import { chartPalette } from '../charts/echarts';
import { useChart } from '../charts/useChart';

/** A shaped `.Curve` of arity 3 (`[x, expiry, value]`) or 4 (`[x, expiry, tenor, value]`) as a
 * heatmap - the widget token folds Surface and Space into one, so ARITY is the branch, and an
 * arity-4 space picks one tenor slice with a select. A heatmap over 3-D eye candy is deliberate:
 * levels read off a colour field, and nothing hides behind a hill. */
export function SurfaceHeatmap({ data }: { data: number[][] }) {
  const arity = data[0]?.length ?? 0;
  const tenors = useMemo(
    () => (arity === 4 ? [...new Set(data.map((row) => row[2]))].sort((a, b) => a - b) : []),
    [data, arity]);
  const [tenor, setTenor] = useState<number | null>(null);
  const slice = arity === 4
    ? data.filter((row) => row[2] === (tenor ?? tenors[0]))
    : data;
  const valueAt = arity === 4 ? 3 : 2;

  const option = useMemo(() => {
    const palette = chartPalette();
    const xs = [...new Set(slice.map((row) => row[0]))].sort((a, b) => a - b);
    const ys = [...new Set(slice.map((row) => row[1]))].sort((a, b) => a - b);
    const values = slice.map((row) => row[valueAt]);
    return {
      animation: false,
      grid: { left: 70, right: 90, top: 20, bottom: 40 },
      tooltip: {},
      xAxis: {
        type: 'category' as const, data: xs.map(String), name: 'moneyness',
        axisLabel: { color: palette.text }, axisLine: { lineStyle: { color: palette.axis } },
      },
      yAxis: {
        type: 'category' as const, data: ys.map(String), name: 'expiry',
        axisLabel: { color: palette.text }, axisLine: { lineStyle: { color: palette.axis } },
      },
      visualMap: {
        min: Math.min(...values), max: Math.max(...values),
        calculable: true, orient: 'vertical' as const, right: 0, top: 'center',
        textStyle: { color: palette.text },
        inRange: { color: ['#3457d5', '#5b8def', '#8fd0a9', '#f2d16b', '#e2704a'] },
      },
      series: [{
        type: 'heatmap' as const,
        data: slice.map((row) => [xs.indexOf(row[0]), ys.indexOf(row[1]), row[valueAt]]),
        label: { show: slice.length <= 60, color: palette.text, fontSize: 10 },
      }],
    };
  }, [slice, valueAt]);

  return (
    <div>
      {arity === 4 && (
        <div className="pager">
          <span>tenor</span>
          <select value={tenor ?? tenors[0]} onChange={(e) => setTenor(Number(e.target.value))}>
            {tenors.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      )}
      <div className="chart tall" ref={useChart(option)} />
    </div>
  );
}
