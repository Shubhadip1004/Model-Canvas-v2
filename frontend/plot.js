function initAllPlots() {
  Plotly.newPlot(
    'boundaryPlot',
    [{ x: [], y: [], mode: 'markers' }],
    {
      margin: { t: 30, b: 40, l: 40, r: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    },
    { responsive: true }
  );

  // Accuracy plot with fixed green color
  Plotly.newPlot(
    'accPlot',
    [{ 
      x: [], 
      y: [], 
      mode: 'lines+markers', 
      name: 'Accuracy',
      line: { color: '#10b981', width: 2 },
      marker: { color: '#10b981', size: 3 }
    }],
    {
      margin: { t: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: { range: [-0.05, 1.05] }
    },
    { responsive: true }
  );

  // Loss plot with fixed red color
  Plotly.newPlot(
    'lossPlot',
    [{ 
      x: [], 
      y: [], 
      mode: 'lines+markers', 
      name: 'Loss',
      line: { color: '#ef4444', width: 2 },
      marker: { color: '#ef4444', size: 3 }
    }],
    {
      margin: { t: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: { rangemode: 'tozero' }
    },
    { responsive: true }
  );

  Plotly.newPlot(
    'confMatrix',
    [{
      z: [[0]],
      x: ['0'],
      y: ['0'],
      type: 'heatmap',
      colorscale: 'Blues',
      showscale: true,
      text: [['0']],
      texttemplate: '%{text}',
      textfont: { color: 'white' }
    }],
    {
      margin: { t: 40, l: 70, b: 40, r: 10 },
      xaxis: { title: 'Prediction' },
      yaxis: { title: 'Actual', autorange: 'reversed' }
    },
    { responsive: true, autosize: true }
  );
}

initAllPlots();

window.renderBoundaryFrame = function (contour, traces, layout, featureView) {
  if (contour && contour.x && contour.y && contour.z) {
    Plotly.react('boundaryPlot', [contour, ...traces], layout, { responsive: true });
  } else {
    Plotly.react('boundaryPlot', [...traces], layout, { responsive: true });
  }
};

// Function to reset metrics plots with fixed colors
window.resetMetricsPlots = function() {
  Plotly.react('accPlot',
    [{ 
      x: [], 
      y: [], 
      mode: 'lines+markers', 
      name: 'Accuracy',
      line: { color: '#10b981', width: 2 },
      marker: { color: '#10b981', size: 3 }
    }],
    {
      margin: { t: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: { range: [-0.05, 1.05] }
    }
  );
  
  Plotly.react('lossPlot',
    [{ 
      x: [], 
      y: [], 
      mode: 'lines+markers', 
      name: 'Loss',
      line: { color: '#ef4444', width: 2 },
      marker: { color: '#ef4444', size: 3 }
    }],
    {
      margin: { t: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      yaxis: { rangemode: 'tozero' }
    }
  );
};