import { useMemo } from 'react';
import { SurfaceChart } from 'echarts-gl/charts';
import { Grid3DComponent } from 'echarts-gl/components';
import echarts, { GRADIENT, chartPalette } from '../charts/echarts';
import { useChart } from '../charts/useChart';
import { tenorLabel, type Mesh } from '../vols';

// Registered onto the same core the other charts use, on IMPORT - and this module is reached only
// through a dynamic import, so the GL code is its own chunk and a desk that never opens the mesh
// never downloads it.
echarts.use([SurfaceChart, Grid3DComponent]);

/** The surface as the notebook drew it: a mesh over expiry and x, coloured by height, rotatable
 * with the mouse. `dataShape` is the grid's own, so the rows are the expiries. */
export default function SurfaceMesh({ mesh, axes }: { mesh: Mesh; axes: string[] }) {
  const option = useMemo(() => {
    const palette = chartPalette();
    const values = mesh.z.flat();
    const axis3D = (name: string, formatter?: unknown) => ({
      type: 'value' as const, name,
      nameTextStyle: { color: palette.text },
      axisLabel: { color: palette.text, formatter },
      axisLine: { lineStyle: { color: palette.axis } },
      splitLine: { lineStyle: { color: palette.axis, opacity: 0.6 } },
    });
    return {
      animation: false,
      tooltip: { formatter: (point: { value: number[] }) =>
        `${axes[1]} ${tenorLabel(point.value[1])}<br/>${axes[0]} ${point.value[0]}<br/>` +
        `${axes[2]} ${(point.value[2] * 100).toFixed(2)} %` },
      visualMap: {
        min: Math.min(...values), max: Math.max(...values), dimension: 2,
        calculable: true, orient: 'vertical' as const, right: 0, top: 'center',
        textStyle: { color: palette.text },
        formatter: (value: number) => `${(value * 100).toFixed(2)} %`,
        inRange: { color: GRADIENT },
      },
      xAxis3D: axis3D(axes[0]),
      yAxis3D: axis3D(axes[1], (value: number) => tenorLabel(value)),
      zAxis3D: axis3D(axes[2], (value: number) => `${(value * 100).toFixed(1)}%`),
      grid3D: {
        boxWidth: 110, boxDepth: 70, boxHeight: 60,
        environment: 'transparent',
        axisPointer: { lineStyle: { color: palette.text } },
        viewControl: { alpha: 22, beta: 35, distance: 190, autoRotate: false },
        light: { main: { intensity: 1.2, shadow: false }, ambient: { intensity: 0.35 } },
      },
      series: [{
        type: 'surface' as const,
        shading: 'lambert' as const,          // the library's own: the lit mesh is what shows shape
        wireframe: { show: true, lineStyle: { color: palette.axis, width: 1 } },
        dataShape: [mesh.expiries.length, mesh.xs.length],
        data: mesh.expiries.flatMap((expiry, row) =>
          mesh.xs.map((x, column) => [x, expiry, mesh.z[row][column]])),
      }],
    };
  }, [mesh, axes]);

  return <div className="chart tall" ref={useChart(option)} />;
}
