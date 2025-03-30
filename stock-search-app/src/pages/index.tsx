import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { StockData, StockOption, PortfolioOptimizationResponse, OptimizationResult } from '../types';
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

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip);

// Component to display base64-encoded images
const ImageComponent: React.FC<{ base64String: string; altText: string }> = ({ base64String, altText }) => (
  <div className="image-container">
    {base64String ? (
      <img
        src={`data:image/png;base64,${base64String}`}
        alt={altText}
        style={{ width: '200%', maxWidth: '1000px', height: 'auto' }}
      />
    ) : (
      <p>No image available</p>
    )}
  </div>
);

// Algorithm options for selection with full descriptions
const algorithmOptions = [
  { label: 'Mean-Variance Optimization', value: 'MVO' },
  { label: 'Minimum Volatility', value: 'MinVol' },
  { label: 'Maximum Quadratic Utility', value: 'MaxQuadraticUtility' },
  { label: 'Equally Weighted', value: 'EquiWeighted' },
  { label: 'Critical Line Algorithm', value: 'CriticalLineAlgorithm' },
  { label: 'Hierarchical Risk Parity (HRP)', value: 'HRP' },
];

// CLA sub-method options
const claSubOptions = [
  { label: 'Mean-Variance Optimization', value: 'MVO' },
  { label: 'Minimum Volatility', value: 'MinVol' },
  { label: 'Both', value: 'Both' },
];

// Mapping for full descriptive headings in the results section
const algoDisplayNames: { [key: string]: string } = {
  MVO: "Mean-Variance Optimization",
  MinVol: "Minimum Volatility",
  MaxQuadraticUtility: "Maximum Quadratic Utility",
  EquiWeighted: "Equally Weighted",
  CriticalLineAlgorithm_MVO: "Critical Line Algorithm (Mean-Variance Optimization)",
  CriticalLineAlgorithm_MinVol: "Critical Line Algorithm (Minimum Volatility)",
  HRP: "Hierarchical Risk Parity (HRP)",
};

const HomePage: React.FC = () => {
  const [stockData, setStockData] = useState<StockData>({});
  const [options, setOptions] = useState<StockOption[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedStocks, setSelectedStocks] = useState<StockOption[]>([]);
  const [optimizationResult, setOptimizationResult] = useState<PortfolioOptimizationResponse | null>(null);
  const [filteredOptions, setFilteredOptions] = useState<StockOption[]>([]);
  // For algorithm selection (controlled)
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<{ label: string; value: string }[]>([]);
  // For CLA sub-method (controlled)
  const [selectedCLA, setSelectedCLA] = useState<{ label: string; value: string }>(claSubOptions[2]); // default = "Both"
  // Loading state for API request
  const [loading, setLoading] = useState(false);

  // Compute whether we can submit
  const canSubmit = selectedStocks.length >= 2 && selectedAlgorithms.length >= 1;
  let submitError = '';
  if (selectedStocks.length < 2 && selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 2 stocks and 1 optimization method.';
  } else if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks.';
  } else if (selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 1 optimization method.';
  }

  // Fetch stock data
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const res = await fetch('/stock_data.json');
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
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

  // Fuzzy search with Fuse.js
  const fuse = useMemo(() => {
    return new Fuse(options, {
      keys: ['ticker', 'name'],
      threshold: 0.3,
    });
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

  // Add/remove stocks
  const handleAddStock = (event: any, newValue: StockOption | null) => {
    if (newValue) {
      const duplicate = selectedStocks.some(
        (s) => s.ticker === newValue.ticker && s.exchange === newValue.exchange
      );
      if (!duplicate) {
        setSelectedStocks([...selectedStocks, newValue]);
      }
    }
    setInputValue('');
  };

  const handleRemoveStock = (stockToRemove: StockOption) => {
    setSelectedStocks(selectedStocks.filter((s) => !(s.ticker === stockToRemove.ticker && s.exchange === stockToRemove.exchange)));
  };

  // Handle algorithm selection (controlled)
  const handleAlgorithmChange = (event: any, newValue: { label: string; value: string }[]) => {
    setSelectedAlgorithms(newValue);
  };

  // Reset CLA sub-method if Critical Line Algorithm is removed
  useEffect(() => {
    if (!selectedAlgorithms.find((algo) => algo.value === 'CriticalLineAlgorithm')) {
      setSelectedCLA(claSubOptions[2]);
    }
  }, [selectedAlgorithms]);

  const handleCLAChange = (event: SelectChangeEvent<string>) => {
    const chosen = claSubOptions.find((opt) => opt.value === event.target.value);
    if (chosen) setSelectedCLA(chosen);
  };

  // Submit data to backend with loading indicator
  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true); // Start loading
    const dataToSend = {
      stocks: selectedStocks.map((s) => ({ ticker: s.ticker, exchange: s.exchange })),
      methods: selectedAlgorithms.map((a) => a.value),
      ...(selectedAlgorithms.some((a) => a.value === 'CriticalLineAlgorithm')
        ? { cla_method: selectedCLA.value }
        : {}),
    };
    // aws link https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/optimize
    try {
      const response = await axios.post('https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/optimize', dataToSend);
      console.log('Backend response:', response.data);
      const result = response.data as PortfolioOptimizationResponse;
      setOptimizationResult(result);
    } catch (error) {
      console.error('API Error:', error);
    } finally {
      setLoading(false); // Stop loading regardless of success or error
    }
  };

  // Reset all selections
  const handleReset = () => {
    setSelectedStocks([]);
    setInputValue('');
    setSelectedAlgorithms([]);
    setSelectedCLA(claSubOptions[2]);
    setOptimizationResult(null);
  };

  // Format date
  const formatDate = (dateStr: string) => {
    const opts: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateStr).toLocaleDateString(undefined, opts);
  };

  // Prepare data for Chart.js
  const prepareChartData = (res: PortfolioOptimizationResponse) => {
    const labels = res.dates.map((d) => new Date(d).toLocaleDateString());
    const datasets = [];
    if (res.cumulative_returns.MVO?.length) {
      datasets.push({
        label: 'Mean-Variance Optimization',
        data: res.cumulative_returns.MVO,
        borderColor: 'blue',
        fill: false,
      });
    }
    if (res.cumulative_returns.MinVol?.length) {
      datasets.push({
        label: 'Minimum Volatility',
        data: res.cumulative_returns.MinVol,
        borderColor: 'green',
        fill: false,
      });
    }
    if (res.cumulative_returns.MaxQuadraticUtility?.length) {
      datasets.push({
        label: 'Max Quadratic Utility',
        data: res.cumulative_returns.MaxQuadraticUtility,
        borderColor: 'purple',
        fill: false,
      });
    }
    if (res.cumulative_returns.EquiWeighted?.length) {
      datasets.push({
        label: 'Equally Weighted',
        data: res.cumulative_returns.EquiWeighted,
        borderColor: 'orange',
        fill: false,
      });
    }
    if (res.cumulative_returns.CriticalLineAlgorithm_MVO?.length) {
      datasets.push({
        label: 'CLA (MVO)',
        data: res.cumulative_returns.CriticalLineAlgorithm_MVO,
        borderColor: 'magenta',
        fill: false,
      });
    }
    if (res.cumulative_returns.CriticalLineAlgorithm_MinVol?.length) {
      datasets.push({
        label: 'CLA (MinVol)',
        data: res.cumulative_returns.CriticalLineAlgorithm_MinVol,
        borderColor: 'teal',
        fill: false,
      });
    }
    if (res.cumulative_returns.HRP?.length) {
      datasets.push({
        label: 'Hierarchical Risk Parity (HRP)',
        data: res.cumulative_returns.HRP,
        borderColor: 'brown',
        fill: false,
      });
    }
    if (res.nifty_returns?.length) {
      datasets.push({
        label: 'Nifty Index',
        data: res.nifty_returns,
        borderColor: 'red',
        fill: false,
      });
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

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-center">Indian Stock Portfolio Optimization</h1>

      {/* Stock Search Section */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Search and Select Stocks</h2>
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
          <h3 className="text-lg font-medium mb-2">Selected Stocks</h3>
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
        <h2 className="text-xl font-semibold mb-2">Select Optimization Algorithms</h2>
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
          <p style={{ color: 'red', marginTop: '0.5rem' }}>{submitError}</p>
        )}
      </div>

      {/* Loading Spinner and Optimization Results */}
      {loading ? (
  <div
    style={{
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '200px',
    }}
  >
    <CircularProgress />
    <div style={{ marginTop: '16px', fontSize: '1.2rem', fontWeight: 'bold' }}>
      Running Optimizations
    </div>
  </div>
) : (
        optimizationResult && (
          <div className="results-container">
            <h2 className="text-2xl font-bold mb-4">Optimization Results</h2>
            <p>
              Data Time Period: {formatDate(optimizationResult.start_date)} to {formatDate(optimizationResult.end_date)}
            </p>
            {Object.entries(optimizationResult.results || {}).map(([methodKey, methodData]) => {
              if (!methodData) return null;
              const perf = methodData.performance;
              return (
                <div key={methodKey} className="result-with-plot">
                  <div className="result-details">
                    <h3 className="text-xl font-semibold">
                      {algoDisplayNames[methodKey] || methodKey} Results
                    </h3>
                    <p>Expected Return: {(perf.expected_return * 100).toFixed(2)}%</p>
                    <p>Volatility: {(perf.volatility * 100).toFixed(2)}%</p>
                    <p>Sharpe Ratio: {perf.sharpe.toFixed(4)}</p>
                    <p>Sortino Ratio: {perf.sortino.toFixed(4)}</p>
                    <p>Max Drawdown: {(perf.max_drawdown * 100).toFixed(2)}%</p>
                    <p>RoMaD: {perf.romad.toFixed(4)}</p>
                    <p>VaR 95%: {(perf.var_95 * 100).toFixed(2)}%</p>
                    <p>CVaR 95%: {(perf.cvar_95 * 100).toFixed(2)}%</p>
                    <p>VaR 90%: {(perf.var_90 * 100).toFixed(2)}%</p>
                    <p>CVaR 90%: {(perf.cvar_90 * 100).toFixed(2)}%</p>
                    <p>CAGR: {(perf.cagr * 100).toFixed(2)}%</p>
                    <p>Portfolio Beta: {perf.portfolio_beta.toFixed(4)}</p>
                    <h4 className="font-semibold mt-2">Weights:</h4>
                    <ul>
                      {Object.entries(methodData.weights).map(([ticker, weight]) => (
                        <li key={ticker}>
                          {ticker}: {(weight * 100).toFixed(2)}%
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="plots-container">
                    <ImageComponent base64String={methodData.returns_dist || ''} altText={`${methodKey} Distribution`} />
                    <ImageComponent base64String={methodData.max_drawdown_plot || ''} altText={`${methodKey} Drawdown`} />
                  </div>
                </div>
              );
            })}

            {/* Cumulative Returns Chart */}
            <div className="mt-6">
              <h2 className="text-2xl font-bold mb-4">Cumulative Returns Over Time</h2>
              <Line data={prepareChartData(optimizationResult)} options={chartOptions} />
            </div>
          </div>
        )
      )}

      <style jsx>{`
        .image-container {
          border: 1px solid #ccc;
          padding: 10px;
          background-color: #f9f9f9;
        }
        .results-container {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }
        .result-with-plot {
          display: flex;
          flex-wrap: wrap;
          gap: 2rem;
          margin-top: 1rem;
          border-bottom: 1px solid #e0e0e0;
          padding-bottom: 1rem;
        }
        .result-details {
          flex: 1;
          min-width: 280px;
        }
        .plots-container {
          flex: 1;
          min-width: 280px;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
      `}</style>
    </div>
  );
};

export default HomePage;
