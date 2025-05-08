import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { StockData, StockOption, PortfolioOptimizationResponse, OptimizationResult, APIError } from '../types';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import CircularProgress from '@mui/material/CircularProgress';
import axios from 'axios';
import Fuse from 'fuse.js';
import debounce from 'lodash/debounce';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
  Tooltip,
} from 'chart.js';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';

// Material UI components
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import GetApp from '@mui/icons-material/GetApp';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip);

// Updated ImageComponent for displaying images
const ImageComponent: React.FC<{ base64String: string; altText: string }> = ({ base64String, altText }) => (
  <div style={{ marginBottom: '1rem' }}>
    {base64String ? (
      <img src={`data:image/png;base64,${base64String}`} alt={altText} style={{ display: 'block' }} />
    ) : (
      <p>No image available</p>
    )}
  </div>
);

// Algorithm options for selection
const algorithmOptions = [
  { label: 'Mean-Variance Optimization', value: 'MVO' },
  { label: 'Minimum Volatility', value: 'MinVol' },
  { label: 'Maximum Quadratic Utility', value: 'MaxQuadraticUtility' },
  { label: 'Equally Weighted', value: 'EquiWeighted' },
  { label: 'Critical Line Algorithm', value: 'CriticalLineAlgorithm' },
  { label: 'Hierarchical Risk Parity (HRP)', value: 'HRP' },
  { label: 'Minimum Conditional Value at Risk (CVaR)', value: 'MinCVaR' },
  { label: 'Minimum Conditional Drawdown at Risk (CDaR)', value: 'MinCDaR' },
  
];

// CLA sub-method options
const claSubOptions = [
  { label: 'Mean-Variance Optimization', value: 'MVO' },
  { label: 'Minimum Volatility', value: 'MinVol' },
  { label: 'Both', value: 'Both' },
];

// Mapping for descriptive headings in the results section
const algoDisplayNames: { [key: string]: string } = {
  MVO: "Mean-Variance Optimization",
  MinVol: "Minimum Volatility",
  MaxQuadraticUtility: "Maximum Quadratic Utility",
  EquiWeighted: "Equally Weighted",
  CriticalLineAlgorithm_MVO: "Critical Line Algorithm (Mean-Variance Optimization)",
  CriticalLineAlgorithm_MinVol: "Critical Line Algorithm (Minimum Volatility)",
  HRP: "Hierarchical Risk Parity (HRP)",
  MinCVaR: "Minimum Conditional Value at Risk (CVaR)",
  MinCDaR: "Minimum Conditional Drawdown at Risk (CDaR)"
};

/**
 * Helper to color-code returns using a gradient:
 * - More positive => more green
 * - More negative => more red
 * - Bold black text
 */
function getReturnCellStyle(ret: number | undefined): React.CSSProperties {
  if (ret === undefined) {
    return { fontWeight: 'bold', color: 'black' };
  }
  const pct = ret * 100;
  const maxMagnitude = 50;
  const intensity = Math.min(Math.abs(pct), maxMagnitude) / maxMagnitude;
  const baseStyle: React.CSSProperties = { fontWeight: 'bold', color: 'black' };
  if (ret >= 0) {
    return { ...baseStyle, backgroundColor: `rgb(0, ${Math.floor(128 + intensity * 80)}, 0)` };
  } else {
    return { ...baseStyle, backgroundColor: `rgb(255, ${Math.floor(50 - intensity * 50)}, ${Math.floor(50 - intensity * 50)})` };
  }
}

const HomePage: React.FC = () => {
  const [stockData, setStockData] = useState<StockData>({});
  const [options, setOptions] = useState<StockOption[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedStocks, setSelectedStocks] = useState<StockOption[]>([]);
  const [optimizationResult, setOptimizationResult] = useState<PortfolioOptimizationResponse | null>(null);
  const [filteredOptions, setFilteredOptions] = useState<StockOption[]>([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<{ label: string; value: string }[]>([]);
  const [selectedCLA, setSelectedCLA] = useState<{ label: string; value: string }>(claSubOptions[2]); // Default "Both"
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<APIError | null>(null);

  const canSubmit = selectedStocks.length >= 2 && selectedAlgorithms.length >= 1;
  let submitError = '';
  if (selectedStocks.length < 2 && selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 2 stocks and 1 optimization method.';
  } else if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks.';
  } else if (selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 1 optimization method.';
  }

  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const res = await fetch('/stock_data.json');
        if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
        const data: StockData = await res.json();
        setStockData(data);
        const newOptions: StockOption[] = [];
        for (const [ticker, listings] of Object.entries(data)) {
          const listingsArray = Array.isArray(listings) ? listings : [listings];
          listingsArray.forEach((listing) => {
            newOptions.push({
              ticker,
              name: listing.name,
              exchange: listing.exchange,
            });
          });
        }
        setOptions(newOptions);
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    };
    fetchStockData();
  }, []);

  const fuse = useMemo(() => {
    return new Fuse(options, { keys: ['ticker', 'name'], threshold: 0.3 });
  }, [options]);

  const debouncedFilter = useCallback(
    debounce((input: string) => {
      if (!input) {
        setFilteredOptions([]);
        return;
      }
      const results = fuse.search(input);
      setFilteredOptions(results.slice(0, 50).map((res) => res.item));
    }, 300),
    [fuse]
  );

  useEffect(() => {
    debouncedFilter(inputValue);
  }, [inputValue, debouncedFilter]);

  const handleAddStock = (event: any, newValue: StockOption | null) => {
    if (newValue) {
      const duplicate = selectedStocks.some(
        (s) => s.ticker === newValue.ticker && s.exchange === newValue.exchange
      );
      if (!duplicate) setSelectedStocks([...selectedStocks, newValue]);
    }
    setInputValue('');
  };

  const handleRemoveStock = (stockToRemove: StockOption) => {
    setSelectedStocks(selectedStocks.filter((s) => !(s.ticker === stockToRemove.ticker && s.exchange === stockToRemove.exchange)));
  };

  const handleAlgorithmChange = (event: any, newValue: { label: string; value: string }[]) => {
    setSelectedAlgorithms(newValue);
  };

  useEffect(() => {
    if (!selectedAlgorithms.find((algo) => algo.value === 'CriticalLineAlgorithm')) {
      setSelectedCLA(claSubOptions[2]);
    }
  }, [selectedAlgorithms]);

  const handleCLAChange = (event: SelectChangeEvent<string>) => {
    const chosen = claSubOptions.find((opt) => opt.value === event.target.value);
    if (chosen) setSelectedCLA(chosen);
  };

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setError(null); // Clear any previous errors
    const dataToSend = {
      stocks: selectedStocks.map((s) => ({ ticker: s.ticker, exchange: s.exchange })),
      methods: selectedAlgorithms.map((a) => a.value),
      ...(selectedAlgorithms.some((a) => a.value === 'CriticalLineAlgorithm')
        ? { cla_method: selectedCLA.value }
        : {}),
    };
    //
    try {
      const response = await axios.post('https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/optimize', dataToSend);
      console.log('Backend response:', response.data);
      const result = response.data as PortfolioOptimizationResponse;
      setOptimizationResult(result);
      
      // Check for warnings in the response
      if (response.data.warnings) {
        setError({
          message: response.data.warnings.message,
          details: response.data.warnings.failed_methods
        });
      }
    } catch (error) {
      console.error('API Error:', error);
      // Handle axios errors
      if (axios.isAxiosError(error) && error.response) {
        // Extract the error details from the backend response
        const errorData = error.response.data.error;
        if (errorData) {
          setError({
            message: errorData.message || 'An error occurred during optimization',
            details: errorData.details
          });
        } else {
          setError({
            message: `Error ${error.response.status}: ${error.response.statusText}`,
          });
        }
      } else {
        // Handle non-axios errors
        setError({
          message: 'Failed to connect to the optimization service. Please try again later.',
        });
      }
      setOptimizationResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedStocks([]);
    setInputValue('');
    setSelectedAlgorithms([]);
    setSelectedCLA(claSubOptions[2]);
    setOptimizationResult(null);
    setError(null); // Clear any errors on reset
  };

  const formatDate = (dateStr: string) => {
    const opts: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateStr).toLocaleDateString(undefined, opts);
  };

  // Update prepareChartData to include the new MinCVaR key with its own color (cyan)
  const prepareChartData = (res: PortfolioOptimizationResponse) => {
    const labels = res.dates.map((d) => new Date(d).toLocaleDateString());
    const datasets = [];
    if (res.cumulative_returns.MVO?.length) {
      datasets.push({ label: 'Mean-Variance Optimization', data: res.cumulative_returns.MVO, borderColor: 'blue', fill: false });
    }
    if (res.cumulative_returns.MinVol?.length) {
      datasets.push({ label: 'Minimum Volatility', data: res.cumulative_returns.MinVol, borderColor: 'green', fill: false });
    }
    if (res.cumulative_returns.MaxQuadraticUtility?.length) {
      datasets.push({ label: 'Max Quadratic Utility', data: res.cumulative_returns.MaxQuadraticUtility, borderColor: 'purple', fill: false });
    }
    if (res.cumulative_returns.EquiWeighted?.length) {
      datasets.push({ label: 'Equally Weighted', data: res.cumulative_returns.EquiWeighted, borderColor: 'orange', fill: false });
    }
    if (res.cumulative_returns.CriticalLineAlgorithm_MVO?.length) {
      datasets.push({ label: 'CLA (MVO)', data: res.cumulative_returns.CriticalLineAlgorithm_MVO, borderColor: 'magenta', fill: false });
    }
    if (res.cumulative_returns.CriticalLineAlgorithm_MinVol?.length) {
      datasets.push({ label: 'CLA (MinVol)', data: res.cumulative_returns.CriticalLineAlgorithm_MinVol, borderColor: 'teal', fill: false });
    }
    if (res.cumulative_returns.HRP?.length) {
      datasets.push({ label: 'Hierarchical Risk Parity (HRP)', data: res.cumulative_returns.HRP, borderColor: 'brown', fill: false });
    }
    // New: Include Minimum CVaR (MinCVaR) with cyan color
    if (res.cumulative_returns.MinCVaR?.length) {
      datasets.push({ label: 'Minimum Conditional VaR (MCVar)', data: res.cumulative_returns.MinCVaR, borderColor: 'cyan', fill: false });
    }
    if (res.cumulative_returns.MinCDaR?.length) {
      datasets.push({ label: 'Minimum Conditional Drawdown at Risk (CDaR)', data: res.cumulative_returns.MinCDaR, borderColor: 'purple', fill: false });
    }
    if (res.nifty_returns?.length) {
      datasets.push({ label: 'Nifty Index', data: res.cumulative_returns.nifty_returns || res.nifty_returns, borderColor: 'red', fill: false });
    }
    return { labels, datasets };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' as const },
      tooltip: { mode: 'index' as const, intersect: false },
    },
    scales: {
      x: { display: true, title: { display: true, text: 'Date' } },
      y: { display: true, title: { display: true, text: 'Cumulative Return' } },
    },
  };

  const allYears = useMemo(() => {
    if (!optimizationResult || !optimizationResult.stock_yearly_returns) return [];
    const yearSet = new Set<string>();
    Object.values(optimizationResult.stock_yearly_returns).forEach((yearData) => {
      Object.keys(yearData).forEach((year) => yearSet.add(year));
    });
    return Array.from(yearSet).sort();
  }, [optimizationResult]);

  // Cache for storing generated PDFs to avoid regeneration
  const [pdfCache, setPdfCache] = useState<{[key: string]: string}>({});
  
  // Canvas cache to eliminate redundant html2canvas calls
  const canvasCache = useRef<Record<string, HTMLCanvasElement>>({});

  // Helper function to render an element to canvas once and cache the result
  const renderOnce = async (
    el: HTMLElement,
    key: string,
    options: any = {
      scale: 1.5,
      useCORS: true,
      logging: false,
      allowTaint: true,
      backgroundColor: null,
      imageTimeout: 0
    }
  ): Promise<HTMLCanvasElement> => {
    if (!canvasCache.current[key]) {
      console.log(`Rendering ${key} to canvas for the first time`);
      canvasCache.current[key] = await html2canvas(el, options);
    } else {
      console.log(`Using cached canvas for ${key}`);
    }
    return canvasCache.current[key];
  };

  // Clear canvas cache when optimization results change
  useEffect(() => {
    if (optimizationResult) {
      canvasCache.current = {};
      console.log('Canvas cache cleared due to new optimization results');
    }
  }, [optimizationResult]);

  const generatePDF = async () => {
    if (!optimizationResult) return;

    // Create a cache key based on selected stocks and optimization result
    const cacheKey = selectedStocks.map(s => s.ticker).join('-') + '-' + 
                     (optimizationResult.start_date || '') + '-' + 
                     (optimizationResult.end_date || '');

    // Check if we already have this PDF cached
    if (pdfCache[cacheKey]) {
      // Use the cached PDF
      const link = document.createElement('a');
      link.href = pdfCache[cacheKey];
      link.download = `portfolio_optimization_${new Date().toISOString().split('T')[0]}.pdf`;
      link.click();
      return;
    }

    // Start timing PDF generation
    const startTime = performance.now();
    console.log('Starting PDF generation...');

    try {
      // Create loading overlay
      const loadingElement = document.createElement('div');
      loadingElement.setAttribute('id', 'pdf-loading-overlay'); // Add an ID for easier removal
      loadingElement.style.position = 'fixed';
      loadingElement.style.top = '0';
      loadingElement.style.left = '0';
      loadingElement.style.width = '100%';
      loadingElement.style.height = '100%';
      loadingElement.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
      loadingElement.style.display = 'flex';
      loadingElement.style.justifyContent = 'center';
      loadingElement.style.alignItems = 'center';
      loadingElement.style.zIndex = '1000';
      loadingElement.innerHTML = '<div style="text-align: center;"><div style="font-size: 24px; margin-bottom: 10px;">Generating PDF...</div><div style="font-size: 14px;">This may take a few seconds</div></div>';
      document.body.appendChild(loadingElement);
      
      // Calculate when the PDF was generated with proper formatting
      const now = new Date();
      const dateOptions: Intl.DateTimeFormatOptions = { 
        year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: true
      };
      const formattedDateTime = now.toLocaleString(undefined, dateOptions);

      // Create a new PDF with A4 size
      const pdfDoc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      // Get page dimensions
      const pageWidth = pdfDoc.internal.pageSize.getWidth();
      const pageHeight = pdfDoc.internal.pageSize.getHeight();
      const margin = 10; // 10mm margin

      // ----------- 1. TITLE PAGE -----------
      pdfDoc.setFontSize(24);
      pdfDoc.text('Portfolio Optimization Report', pageWidth / 2, 40, { align: 'center' });
      
      pdfDoc.setFontSize(12);
      pdfDoc.text(`Generated on: ${formattedDateTime}`, pageWidth / 2, 50, { align: 'center' });
      
      pdfDoc.setFontSize(14);
      pdfDoc.text('Selected Stocks:', pageWidth / 2, 70, { align: 'center' });
      
      // Add selected stocks in a formatted way
      const stockList = selectedStocks.map(s => `${s.ticker} (${s.exchange})`).join(', ');
      pdfDoc.setFontSize(12);
      
      // Handle long stock lists by wrapping text
      const splitStocks = pdfDoc.splitTextToSize(stockList, pageWidth - (margin * 2));
      pdfDoc.text(splitStocks, pageWidth / 2, 80, { align: 'center' });
      
      // Add period and risk-free rate
      pdfDoc.setFontSize(12);
      pdfDoc.text(`Time Period: ${formatDate(optimizationResult.start_date)} to ${formatDate(optimizationResult.end_date)}`, pageWidth / 2, 100, { align: 'center' });
      pdfDoc.text(`Risk-free Rate: ${(optimizationResult.risk_free_rate! * 100).toFixed(4)}%`, pageWidth / 2, 110, { align: 'center' });

      // ----------- 2. OPTIMIZATION METHOD RESULTS -----------
      const resultsContainer = document.getElementById('optimization-results');
      if (resultsContainer) {
        const methodCards = resultsContainer.querySelectorAll('.method-card');
        
        // Extract method keys for proper caching
        const methodKeys: string[] = [];
        methodCards.forEach(card => {
          const nameElement = card.querySelector('h5');
          const displayName = nameElement ? nameElement.textContent || '' : '';
          let methodKey = '';
          
          // Find the matching method key in algoDisplayNames
          for (const [key, name] of Object.entries(algoDisplayNames)) {
            if (displayName.includes(name)) {
              methodKey = key;
              break;
            }
          }
          
          methodKeys.push(methodKey || `unknown-${Math.random().toString(36).substring(2, 7)}`);
        });
        
        // Process each method card
        for (let i = 0; i < methodCards.length; i++) {
          const card = methodCards[i] as HTMLElement;
          const methodKey = methodKeys[i];
          pdfDoc.addPage();
          
          // Get the method name
          const methodNameElement = card.querySelector('h5');
          const methodName = methodNameElement ? methodNameElement.textContent || 'Optimization Method' : 'Optimization Method';
          
          pdfDoc.setFontSize(16);
          pdfDoc.text(methodName, pageWidth / 2, 20, { align: 'center' });
          
          try {
            // Use the cached canvas or generate a new one
            const cacheKey = `card-${methodKey}`;
            const cardCanvas = await renderOnce(card, cacheKey);
            
            // Add image to PDF
            const cardImgData = cardCanvas.toDataURL('image/png');
            const cardImgWidth = pageWidth - (margin * 2);
            const cardImgHeight = (cardCanvas.height * cardImgWidth) / cardCanvas.width;
            
            // Scale image if too large
            if (cardImgHeight > pageHeight - 30) {
              const scaleFactor = (pageHeight - 30) / cardImgHeight;
              const adjustedWidth = cardImgWidth * scaleFactor;
              const adjustedHeight = cardImgHeight * scaleFactor;
              pdfDoc.addImage(cardImgData, 'PNG', margin, 30, adjustedWidth, adjustedHeight);
            } else {
              pdfDoc.addImage(cardImgData, 'PNG', margin, 30, cardImgWidth, cardImgHeight);
            }
          } catch (err) {
            console.error(`Failed to render card for ${methodName}:`, err);
            pdfDoc.setFontSize(12);
            pdfDoc.text(`Unable to render optimization result for ${methodName}.`, margin, 40);
          }
        }
      }

      // ----------- 3. CUMULATIVE RETURNS CHART -----------
      pdfDoc.addPage();
      pdfDoc.setFontSize(16);
      pdfDoc.text('Cumulative Returns Over Time', pageWidth / 2, 20, { align: 'center' });
      
      // Use existing chart element with caching
      const chartElement = document.querySelector('#cumulative-returns-chart canvas');
      if (chartElement) {
        try {
          // Use the cached canvas or generate a new one
          const chartCanvas = await renderOnce(chartElement as HTMLElement, 'cumulative-returns-chart');
          
          const chartImgData = chartCanvas.toDataURL('image/png');
          const chartImgWidth = pageWidth - (margin * 2);
          const chartImgHeight = (chartCanvas.height * chartImgWidth) / chartCanvas.width;
          
          pdfDoc.addImage(chartImgData, 'PNG', margin, 30, chartImgWidth, chartImgHeight);
        } catch (err) {
          console.error('Failed to render cumulative returns chart:', err);
          pdfDoc.setFontSize(12);
          pdfDoc.text('Unable to render cumulative returns chart.', margin, 40);
        }
      }

      // ----------- 4. YEARLY RETURNS TABLE -----------
      const yearlyReturnsSection = document.querySelector('.yearly-returns-section');
      if (yearlyReturnsSection) {
        pdfDoc.addPage();
        pdfDoc.setFontSize(16);
        pdfDoc.text('Yearly Stock Returns', pageWidth / 2, 20, { align: 'center' });
        
        try {
          // Use the cached canvas or generate a new one
          const tableCanvas = await renderOnce(yearlyReturnsSection as HTMLElement, 'yearly-returns-table');
          
          const tableImgData = tableCanvas.toDataURL('image/png');
          const tableImgWidth = pageWidth - (margin * 2);
          const tableImgHeight = (tableCanvas.height * tableImgWidth) / tableCanvas.width;
          
          // Scale image if too large
          if (tableImgHeight > pageHeight - 30) {
            const scaleFactor = (pageHeight - 30) / tableImgHeight;
            pdfDoc.addImage(
              tableImgData, 
              'PNG', 
              margin, 
              30, 
              tableImgWidth * scaleFactor, 
              tableImgHeight * scaleFactor
            );
          } else {
            pdfDoc.addImage(tableImgData, 'PNG', margin, 30, tableImgWidth, tableImgHeight);
          }
        } catch (err) {
          console.error('Error rendering yearly returns table:', err);
          // Add error message to PDF instead
          pdfDoc.text('Unable to render yearly returns table. The data may be too large.', margin, 40);
        }
      }

      // ----------- 5. COVARIANCE HEATMAP -----------
      if (optimizationResult.covariance_heatmap) {
        pdfDoc.addPage();
        pdfDoc.setFontSize(16);
        pdfDoc.text('Variance-Covariance Matrix', pageWidth / 2, 20, { align: 'center' });
        
        // Use the base64 data directly (no need for html2canvas)
        const covImgWidth = pageWidth - (margin * 2);
        const covImgHeight = covImgWidth * 0.8; // Approximate aspect ratio
        
        try {
          pdfDoc.addImage(
            `data:image/png;base64,${optimizationResult.covariance_heatmap}`, 
            'PNG', 
            margin, 
            30, 
            covImgWidth, 
            covImgHeight
          );
        } catch (err) {
          console.error('Failed to add covariance heatmap to PDF:', err);
          pdfDoc.setFontSize(12);
          pdfDoc.text('Unable to render variance-covariance matrix.', margin, 40);
        }
      }
      
      // Generate PDF as data URL
      const pdfOutput = pdfDoc.output('dataurlstring');
      
      // Cache the generated PDF for future use
      setPdfCache(prev => ({
        ...prev,
        [cacheKey]: pdfOutput
      }));
      
      // Trigger download through a link element
      const downloadLink = document.createElement('a');
      downloadLink.href = pdfOutput;
      downloadLink.download = `portfolio_optimization_${now.toISOString().split('T')[0]}.pdf`;
      downloadLink.click();
      
      // Log generation time
      const endTime = performance.now();
      console.log(`PDF generation completed in ${(endTime - startTime).toFixed(0)}ms`);
      
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      // Clean up loading overlay - use the ID to ensure it's removed
      const loadingOverlay = document.getElementById('pdf-loading-overlay');
      if (loadingOverlay) {
        loadingOverlay.remove();
      }
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <Typography variant="h3" align="center" gutterBottom>
        Indian Stock Portfolio Optimization
      </Typography>

      {/* Stock Search Section */}
      <div className="mb-6">
        <Typography variant="h5" gutterBottom>
          Search and Select Stocks
        </Typography>
        <Autocomplete
          options={filteredOptions}
          getOptionLabel={(o) => `${o.ticker} - ${o.name} (${o.exchange})`}
          onChange={handleAddStock}
          inputValue={inputValue}
          onInputChange={(e, v) => setInputValue(v)}
          renderInput={(params) => <TextField {...params} label="Search Stock" variant="outlined" />}
          style={{ width: '100%', maxWidth: 600 }}
          openOnFocus={false}
          filterOptions={(x) => x}
          clearOnBlur={false}
          value={null}
        />
        <div className="mt-4">
          <Typography variant="h6">Selected Stocks</Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            {selectedStocks.map((stock, idx) => (
              <Chip
                key={idx}
                label={`${stock.ticker} (${stock.exchange})`}
                onDelete={() => handleRemoveStock(stock)}
                className="m-1"
              />
            ))}
          </Stack>
        </div>
      </div>

      {/* Algorithm Selection Section */}
      <div className="mb-6">
        <Typography variant="h5" gutterBottom>
          Select Optimization Algorithms
        </Typography>
        <Autocomplete
          multiple
          options={algorithmOptions}
          getOptionLabel={(o) => o.label}
          onChange={handleAlgorithmChange}
          value={selectedAlgorithms}
          renderInput={(params) => (
            <TextField {...params} variant="outlined" label="Choose Algorithms" placeholder="Select one or more" />
          )}
          style={{ width: '100%', maxWidth: 600 }}
        />
        
        {/* Display info about MOSEK requirement */}
        {selectedAlgorithms.some(algo => ['MinCVaR', 'MinCDaR'].includes(algo.value)) && (
          <Card style={{ marginTop: '1rem', backgroundColor: '#e8f4fd', maxWidth: 600 }}>
            <CardContent>
              <Typography variant="body2" color="info.dark">
                <strong>Note:</strong> MinCVaR and MinCDaR optimizations require a MOSEK license. 
                If license is not available, the system will fall back to Minimum Volatility optimization.
              </Typography>
            </CardContent>
          </Card>
        )}
        
        {selectedAlgorithms.some((algo) => algo.value === 'CriticalLineAlgorithm') && (
          <div className="mt-4 max-w-sm">
            <FormControl fullWidth variant="outlined">
              <InputLabel id="cla-select-label">Select Critical Line Sub-Method</InputLabel>
              <Select
                labelId="cla-select-label"
                value={selectedCLA.value}
                onChange={handleCLAChange}
                label="Select Critical Line Sub-Method"
              >
                {claSubOptions.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="mb-6">
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          className="mr-4"
          disabled={!canSubmit || loading}
        >
          Submit
        </Button>
        <Button variant="outlined" color="secondary" onClick={handleReset}>
          Reset
        </Button>
        {!canSubmit && (
          <Typography color="error" style={{ marginTop: '0.5rem' }}>
            {submitError}
          </Typography>
        )}
      </div>

      {/* Loading Spinner and Optimization Results */}
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: '200px' }}>
          <CircularProgress />
          <div style={{ marginTop: '16px', fontSize: '1.2rem', fontWeight: 'bold' }}>
            Running Optimizations
          </div>
        </div>
      ) : error ? (
        <Card style={{ marginTop: '2rem', backgroundColor: '#ffebee' }}>
          <CardContent>
            <Typography variant="h5" color="error" gutterBottom>
              Optimization Error
            </Typography>
            <Typography variant="body1" gutterBottom>
              {(error as APIError).message}
            </Typography>
            {(error as APIError).details && (
              <div style={{ marginTop: '1rem' }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Details:
                </Typography>
                {Array.isArray((error as APIError).details) ? (
                  <ul>
                    {((error as APIError).details as any[]).map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                ) : typeof (error as APIError).details === 'object' ? (
                  <pre style={{ whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: '200px' }}>
                    {JSON.stringify((error as APIError).details, null, 2)}
                  </pre>
                ) : (
                  <Typography>{String((error as APIError).details)}</Typography>
                )}
              </div>
            )}
            <Button 
              variant="contained" 
              color="primary" 
              style={{ marginTop: '1rem' }}
              onClick={() => setError(null)}
            >
              Dismiss
            </Button>
          </CardContent>
        </Card>
      ) : (
        optimizationResult && (
          <div id="optimization-results" className="results-container">
            <Typography variant="h4" align="center" gutterBottom>
              Optimization Results
            </Typography>
            
            {/* Download Results Button - Moved here for better visibility */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }} data-pdf-exclude="true">
              <Button
                variant="contained"
                color="primary"
                onClick={generatePDF}
                startIcon={<GetApp />}
                size="large"
              >
                Download Results as PDF
              </Button>
            </div>
            
            <Typography variant="body1" align="center">
              Data Time Period: {formatDate(optimizationResult!.start_date)} to {formatDate(optimizationResult!.end_date)} <br></br>
              <div>
                <strong>
                    Benchmark Risk Free Rate (Based on Mean 10-Y GSec yields) : {(optimizationResult!.risk_free_rate! * 100).toFixed(4)}%
                </strong>
              </div>
            </Typography>

            {/* Display warnings about failed methods if present */}
            {error !== null && (
              <Card style={{ marginBottom: '1.5rem', backgroundColor: '#fff3e0', padding: '0.5rem' }}>
                <CardContent>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <Typography variant="subtitle1" color="warning.dark" gutterBottom>
                        <strong>Warning:</strong> {(error as APIError).message}
                      </Typography>
                      {(error as APIError).details && Array.isArray((error as APIError).details) && ((error as APIError).details as string[]).length > 0 && (
                        <>
                          <Typography variant="body2">
                            The following optimization methods failed:
                          </Typography>
                          <ul style={{ margin: '0.5rem 0' }}>
                            {((error as APIError).details as string[]).map((method: string, idx: number) => (
                              <li key={idx}>{algoDisplayNames[method] || method}</li>
                            ))}
                          </ul>
                          <Typography variant="body2">
                            Results from successful methods are still displayed below.
                          </Typography>
                        </>
                      )}
                    </div>
                    <Button 
                      variant="outlined" 
                      color="warning" 
                      size="small"
                      onClick={() => setError(null)}
                      style={{ marginLeft: '1rem', minWidth: 'auto' }}
                    >
                      ✕
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {Object.entries(optimizationResult!.results || {}).map(([methodKey, methodData]) => {
              if (!methodData) return null;
              const perf = methodData.performance;
              return (
                <Card key={methodKey} style={{ marginBottom: '1.5rem' }} className="method-card">
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      {algoDisplayNames[methodKey] || methodKey} Results
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={4}>
                        <Table size="small">
                          <TableBody>
                            <TableRow>
                              <TableCell><strong>Expected Return</strong></TableCell>
                              <TableCell>{(perf.expected_return * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Volatility</strong></TableCell>
                              <TableCell>{(perf.volatility * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Sharpe Ratio</strong></TableCell>
                              <TableCell>{perf.sharpe.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Sortino Ratio</strong></TableCell>
                              <TableCell>{perf.sortino.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Max Drawdown</strong></TableCell>
                              <TableCell>{(perf.max_drawdown * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>RoMaD</strong></TableCell>
                              <TableCell>{perf.romad.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>VaR 95%</strong></TableCell>
                              <TableCell>{(perf.var_95 * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>CVaR 95%</strong></TableCell>
                              <TableCell>{(perf.cvar_95 * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>VaR 90%</strong></TableCell>
                              <TableCell>{(perf.var_90 * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>CVaR 90%</strong></TableCell>
                              <TableCell>{(perf.cvar_90 * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>CAGR</strong></TableCell>
                              <TableCell>{(perf.cagr * 100).toFixed(2)}%</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Portfolio Beta</strong></TableCell>
                              <TableCell>{perf.portfolio_beta.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Skewness</strong></TableCell>
                              <TableCell>{perf.skewness.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Kurtosis</strong></TableCell>
                              <TableCell>{perf.kurtosis.toFixed(4)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>Entropy</strong></TableCell>
                              <TableCell>{perf.entropy.toFixed(4)}</TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>

                        <Typography variant="subtitle1" style={{ marginTop: '1rem', fontWeight: 'bold' }}>
                          Weights
                        </Typography>
                        <Table size="small">
                          <TableBody>
                            {Object.entries(methodData.weights).map(([ticker, weight]) => (
                              <TableRow key={ticker}>
                                <TableCell>{ticker}</TableCell>
                                <TableCell>{(weight * 100).toFixed(2)}%</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Grid>

                      <Grid item xs={12} md={8}>
                        <ImageComponent base64String={methodData.returns_dist || ''} altText={`${methodKey} Distribution`} />
                        <ImageComponent base64String={methodData.max_drawdown_plot || ''} altText={`${methodKey} Drawdown`} />
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              );
            })}

            <div style={{ marginTop: '2rem' }} id="cumulative-returns-chart">
              <Typography variant="h5" align="center" gutterBottom>
                Cumulative Returns Over Time
              </Typography>
              <Line data={prepareChartData(optimizationResult!)} options={chartOptions} />
            </div>

            {optimizationResult!.stock_yearly_returns && (
              <div style={{ marginTop: '2rem' }} className="yearly-returns-section">
                <Typography variant="h5" align="center" gutterBottom>
                  Yearly Stock Returns
                </Typography>
                <Table
                  className="yearly-returns-table"
                  sx={{
                    border: '1px solid black',
                    borderCollapse: 'collapse',
                    '& th, & td': { border: '1px solid black' },
                  }}
                >
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Stock</strong></TableCell>
                      {allYears.map((year) => (
                        <TableCell key={year} align="center"><strong>{year}</strong></TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(optimizationResult!.stock_yearly_returns).map(([ticker, yearData]) => (
                      <TableRow key={ticker}>
                        <TableCell style={{ fontWeight: 'bold', color: 'black' }}>{ticker}</TableCell>
                        {allYears.map((year) => {
                          const ret = yearData[year];
                          return (
                            <TableCell key={year} align="center" style={getReturnCellStyle(ret)}>
                              {ret !== undefined ? (ret * 100).toFixed(2) + '%' : '-'}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}

            {/* Covariance Heatmap Section */}
            {optimizationResult!.covariance_heatmap && (
              <div style={{ marginTop: '2rem', textAlign: 'center' }}>
                <Typography variant="h5" gutterBottom>
                  Variance-Covariance Matrix
                </Typography>
                <div style={{ display: 'inline-block' }}>
                  <ImageComponent base64String={optimizationResult!.covariance_heatmap} altText="Covariance Heatmap" />
                </div>
              </div>
            )}
          </div>
        )
      )}

      <style jsx>{`
        .results-container {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
      `}</style>
    </div>
  );
};

export default HomePage;