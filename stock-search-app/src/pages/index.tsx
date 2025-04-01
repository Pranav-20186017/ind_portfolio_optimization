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

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip);

// Updated ImageComponent for displaying images
const ImageComponent: React.FC<{ base64String: string; altText: string }> = ({ base64String, altText }) => (
  <div style={{ marginBottom: '1rem' }}>
    {base64String ? (
      <img
        src={`data:image/png;base64,${base64String}`}
        alt={altText}
        style={{
          display: 'block',
        }}
      />
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
};

/**
 * Helper to color-code returns using a gradient:
 * - More positive => more green
 * - More negative => more red
 * - Bold black text
 */
function getReturnCellStyle(ret: number | undefined): React.CSSProperties {
  if (ret === undefined) {
    return {
      fontWeight: 'bold',
      color: 'black',
    };
  }
  
  // Convert to percentage
  const pct = ret * 100;
  
  // Lower maxMagnitude to make moderate returns more colorful
  const maxMagnitude = 50;
  
  // Scale the absolute value
  const intensity = Math.min(Math.abs(pct), maxMagnitude) / maxMagnitude;
  
  const baseStyle: React.CSSProperties = {
    fontWeight: 'bold',
    color: 'black', // All text black for consistency
  };

  if (ret >= 0) {
    // Positive => green (darker green for better visibility)
    return {
      ...baseStyle,
      backgroundColor: `rgb(0, ${Math.floor(128 + intensity * 80)}, 0)`,
    };
  } else {
    // Negative => true vibrant red
    return {
      ...baseStyle,
      backgroundColor: `rgb(255, ${Math.floor(50 - intensity * 50)}, ${Math.floor(50 - intensity * 50)})`,
    };
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

  // Check whether the form can be submitted
  const canSubmit = selectedStocks.length >= 2 && selectedAlgorithms.length >= 1;
  let submitError = '';
  if (selectedStocks.length < 2 && selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 2 stocks and 1 optimization method.';
  } else if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks.';
  } else if (selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 1 optimization method.';
  }

  // Fetch stock data from local JSON file
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

  // Add and remove stocks from the selection
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
    setSelectedStocks(
      selectedStocks.filter((s) => !(s.ticker === stockToRemove.ticker && s.exchange === stockToRemove.exchange))
    );
  };

  // Handle algorithm selection (controlled)
  const handleAlgorithmChange = (event: any, newValue: { label: string; value: string }[]) => {
    setSelectedAlgorithms(newValue);
  };

  // Reset CLA sub-method if CLA is removed from algorithms
  useEffect(() => {
    if (!selectedAlgorithms.find((algo) => algo.value === 'CriticalLineAlgorithm')) {
      setSelectedCLA(claSubOptions[2]);
    }
  }, [selectedAlgorithms]);

  const handleCLAChange = (event: SelectChangeEvent<string>) => {
    const chosen = claSubOptions.find((opt) => opt.value === event.target.value);
    if (chosen) setSelectedCLA(chosen);
  };

  // Submit the data to the backend
  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    const dataToSend = {
      stocks: selectedStocks.map((s) => ({ ticker: s.ticker, exchange: s.exchange })),
      methods: selectedAlgorithms.map((a) => a.value),
      ...(selectedAlgorithms.some((a) => a.value === 'CriticalLineAlgorithm')
        ? { cla_method: selectedCLA.value }
        : {}),
    };
    try {
      const response = await axios.post('https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/optimize', dataToSend);
      console.log('Backend response:', response.data);
      const result = response.data as PortfolioOptimizationResponse;
      setOptimizationResult(result);
    } catch (error) {
      console.error('API Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedStocks([]);http://
    setInputValue('');
    setSelectedAlgorithms([]);
    setSelectedCLA(claSubOptions[2]);
    setOptimizationResult(null);
  };

  // Format date string for display
  const formatDate = (dateStr: string) => {
    const opts: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateStr).toLocaleDateString(undefined, opts);
  };

  // Prepare Chart.js data
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

  // Extract unique years from the stock_yearly_returns field for table columns
  const allYears = useMemo(() => {
    if (!optimizationResult || !optimizationResult.stock_yearly_returns) return [];
    const yearSet = new Set<string>();
    Object.values(optimizationResult.stock_yearly_returns).forEach((yearData) => {
      Object.keys(yearData).forEach((year) => yearSet.add(year));
    });
    return Array.from(yearSet).sort();
  }, [optimizationResult]);

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
            <Typography variant="h4" align="center" gutterBottom>
              Optimization Results
            </Typography>
            <Typography variant="body1" align="center">
              Data Time Period: {formatDate(optimizationResult.start_date)} to {formatDate(optimizationResult.end_date)}
            </Typography>

            {Object.entries(optimizationResult.results || {}).map(([methodKey, methodData]) => {
              if (!methodData) return null;
              const perf = methodData.performance;
              return (
                <Card key={methodKey} style={{ marginBottom: '1.5rem' }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      {algoDisplayNames[methodKey] || methodKey} Results
                    </Typography>
                    <Grid container spacing={2}>
                      {/* Left Column: Metrics & Weights */}
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

                      {/* Right Column: Images */}
                      <Grid item xs={12} md={8}>
                        <ImageComponent
                          base64String={methodData.returns_dist || ''}
                          altText={`${methodKey} Distribution`}
                        />
                        <ImageComponent
                          base64String={methodData.max_drawdown_plot || ''}
                          altText={`${methodKey} Drawdown`}
                        />
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              );
            })}

            <div style={{ marginTop: '2rem' }}>
              <Typography variant="h5" align="center" gutterBottom>
                Cumulative Returns Over Time
              </Typography>
              <Line data={prepareChartData(optimizationResult)} options={chartOptions} />
            </div>

            {/* Yearly Returns Matrix */}
            {optimizationResult.stock_yearly_returns && (
              <div style={{ marginTop: '2rem' }}>
                <Typography variant="h5" align="center" gutterBottom>
                  Yearly Stock Returns
                </Typography>
                <Table
                  sx={{
                    border: '1px solid black',
                    borderCollapse: 'collapse',
                    '& th, & td': {
                      border: '1px solid black',
                    },
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
                    {Object.entries(optimizationResult.stock_yearly_returns).map(([ticker, yearData]) => (
                      <TableRow key={ticker}>
                        <TableCell style={{ fontWeight: 'bold', color: 'black' }}>{ticker}</TableCell>
                        {allYears.map((year) => {
                          const ret = yearData[year];
                          return (
                            <TableCell
                              key={year}
                              align="center"
                              style={getReturnCellStyle(ret)}
                            >
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
