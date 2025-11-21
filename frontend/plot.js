// plot.js - initialize canvas and small helpers
function initAllPlots() {
  Plotly.newPlot('boundaryPlot', [{x:[], y:[], mode:'markers'}], {
    margin:{t:30, b:40, l:40, r:10},
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)'
  }, {responsive:true});

  Plotly.newPlot('accPlot', [{x:[], y:[], mode:'lines+markers', name:'Accuracy'}], {
    margin:{t:10}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
  }, {responsive:true});

  Plotly.newPlot('lossPlot', [{x:[], y:[], mode:'lines+markers', name:'Loss'}], {
    margin:{t:10}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)'
  }, {responsive:true});

  Plotly.newPlot('confMatrix', [{z:[[0]], type:'heatmap'}], {margin:{t:10}}, {responsive:true});
}

initAllPlots();

// convenience function exported to app.js
window.renderBoundaryFrame = function(contour, traces, layout) {
  // Reactively update the main plot: show contour + traces
  Plotly.react('boundaryPlot', [contour, ...traces], layout, {responsive:true});
};
