import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  StockData, 
  StockOption, 
  ExchangeEnum,
  DividendOptRequest,
  DividendOptResponse,
  APIError,
  StockItem
} from '../types';
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
import Link from 'next/link';
import axios from 'axios';
import Fuse from 'fuse.js';
import debounce from 'lodash/debounce';
import Head from 'next/head';

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
import InfoOutlined from '@mui/icons-material/InfoOutlined';
import Tooltip from '@mui/material/Tooltip';
import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import InputAdornment from '@mui/material/InputAdornment';

// Import TopNav component
import TopNav from '../components/TopNav';

const DividendOptimizer: React.FC = () => {
  const [stockData, setStockData] = useState<StockData>({});
  const [options, setOptions] = useState<StockOption[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedStocks, setSelectedStocks] = useState<StockOption[]>([]);
  const [filteredOptions, setFilteredOptions] = useState<StockOption[]>([]);
  const [selectedExchange, setSelectedExchange] = useState<ExchangeEnum | null>(null);
  
  // Investment and optimization parameters
  const [budget, setBudget] = useState<string>('1000000'); // Default 10L
  const [entropyWeight, setEntropyWeight] = useState<number>(0.05);
  const [volCap, setVolCap] = useState<number | null>(null);
  const [useVolCap, setUseVolCap] = useState<boolean>(false);
  const [useMedianTtm, setUseMedianTtm] = useState<boolean>(false);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<APIError | null>(null);
  const [optimizationResult, setOptimizationResult] = useState<DividendOptResponse | null>(null);

  // Default high-dividend yield stocks
  const defaultNSEStocks: StockOption[] = [
    { ticker: 'ITC', name: 'ITC Ltd', exchange: 'NSE' },
    { ticker: 'COALINDIA', name: 'Coal India Ltd', exchange: 'NSE' },
    { ticker: 'ONGC', name: 'Oil and Natural Gas Corporation Ltd', exchange: 'NSE' },
    { ticker: 'NTPC', name: 'NTPC Ltd', exchange: 'NSE' },
    { ticker: 'POWERGRID', name: 'Power Grid Corporation of India Ltd', exchange: 'NSE' },
    { ticker: 'IOC', name: 'Indian Oil Corporation Ltd', exchange: 'NSE' },
    { ticker: 'VEDL', name: 'Vedanta Ltd', exchange: 'NSE' },
    { ticker: 'SJVN', name: 'SJVN Ltd', exchange: 'NSE' },
    { ticker: 'NMDC', name: 'NMDC Ltd', exchange: 'NSE' },
    { ticker: 'HINDPETRO', name: 'Hindustan Petroleum Corporation Ltd', exchange: 'NSE' },
    { ticker: 'GAIL', name: 'GAIL India Ltd', exchange: 'NSE' },
    { ticker: 'BPCL', name: 'Bharat Petroleum Corporation Ltd', exchange: 'NSE' },
  ];

  const defaultBSEStocks: StockOption[] = [
    { ticker: 'ITC', name: 'ITC Ltd', exchange: 'BSE' },
    { ticker: 'COALINDIA', name: 'Coal India Ltd', exchange: 'BSE' },
    { ticker: 'ONGC', name: 'Oil and Natural Gas Corporation Ltd', exchange: 'BSE' },
    { ticker: 'NTPC', name: 'NTPC Ltd', exchange: 'BSE' },
    { ticker: 'POWERGRID', name: 'Power Grid Corporation of India Ltd', exchange: 'BSE' },
    { ticker: 'IOC', name: 'Indian Oil Corporation Ltd', exchange: 'BSE' },
    { ticker: 'VEDL', name: 'Vedanta Ltd', exchange: 'BSE' },
    { ticker: 'SJVN', name: 'SJVN Ltd', exchange: 'BSE' },
    { ticker: 'NMDC', name: 'NMDC Ltd', exchange: 'BSE' },
    { ticker: 'HINDPETRO', name: 'Hindustan Petroleum Corporation Ltd', exchange: 'BSE' },
    { ticker: 'GAIL', name: 'GAIL India Ltd', exchange: 'BSE' },
    { ticker: 'BPCL', name: 'Bharat Petroleum Corporation Ltd', exchange: 'BSE' },
  ];

  // Get default stocks based on selected exchange
  const getDefaultStocks = () => {
    if (!selectedExchange) return [];
    return selectedExchange === ExchangeEnum.NSE ? defaultNSEStocks : defaultBSEStocks;
  };

  // Validation
  const canSubmit = selectedStocks.length >= 2 && selectedExchange !== null;

  let submitError = '';
  if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks for diversification.';
  }

  // Load stock data
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

  // Search functionality
  const fuse = useMemo(() => {
    return new Fuse(options, { keys: ['ticker', 'name'], threshold: 0.3 });
  }, [options]);

  const debouncedFilter = useCallback(
    debounce((input: string) => {
      if (!input) {
        setFilteredOptions(getDefaultStocks());
        return;
      }
      const results = fuse.search(input);
      const filteredResults = results
        .map((res) => res.item)
        .filter((item) => !selectedExchange || item.exchange === selectedExchange)
        .slice(0, 50);
      setFilteredOptions(filteredResults);
    }, 300),
    [fuse, selectedExchange]
  );

  useEffect(() => {
    if (!inputValue) {
      setFilteredOptions(getDefaultStocks());
    } else {
      debouncedFilter(inputValue);
    }
  }, [inputValue, debouncedFilter, selectedExchange]);

  useEffect(() => {
    setFilteredOptions(getDefaultStocks());
  }, [selectedExchange]);

  // Stock management
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

  // Submit optimization
  const handleSubmit = async () => {
    if (!canSubmit) return;
    
    setLoading(true);
    setError(null);
    
    const dataToSend: DividendOptRequest = {
      stocks: selectedStocks.map((s) => ({ ticker: s.ticker, exchange: s.exchange as ExchangeEnum })),
      budget: budget ? parseFloat(budget) : undefined,
      entropy_weight: entropyWeight,
      vol_cap: useVolCap && volCap ? volCap : undefined,
      use_median_ttm: useMedianTtm
    };
    
    try {
      const response = await axios.post('https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/dividend-optimize', dataToSend);
      console.log('Backend response:', response.data);
      const result = response.data as DividendOptResponse;
      setOptimizationResult(result);
    } catch (error) {
      console.error('API Error:', error);
      if (axios.isAxiosError(error) && error.response) {
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
        setError({
          message: 'Failed to connect to the optimization service. Please try again later.',
        });
      }
      setOptimizationResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Reset function
  const handleReset = () => {
    setSelectedStocks([]);
    setInputValue('');
    setBudget('1000000');
    setEntropyWeight(0.05);
    setVolCap(null);
    setUseVolCap(false);
    setUseMedianTtm(false);
    setOptimizationResult(null);
    setError(null);
  };

  // Reset when exchange changes
  useEffect(() => {
    setSelectedStocks([]);
    setInputValue('');
    setOptimizationResult(null);
    setError(null);
  }, [selectedExchange]);

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Format currency
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  return (
    <>
      <Head>
        <title>Dividend Yield Optimizer | Entropy-Based Portfolio Optimization</title>
        <meta
          name="description"
          content="Optimize your dividend portfolio using entropy-based diversification. Maximize yield while maintaining portfolio diversity with NSE & BSE stocks."
        />
        <meta name="keywords" content="dividend yield optimization, entropy diversification, Indian dividend stocks, NSE dividend, BSE dividend, portfolio optimization" />
      </Head>
      
      <TopNav />
      
      <Box sx={{ maxWidth: 900, mx: 'auto', mt: 4, mb: 5, px: 2 }}>
        <Typography
          component="h1"
          variant="h4"
          sx={{ fontWeight: 700, mb: 2, color: '#222', textAlign: { xs: 'center', md: 'left' } }}
        >
          Dividend Yield Optimizer
        </Typography>
        <Typography
          variant="body1"
          sx={{
            mb: 3,
            fontSize: '1.1rem',
            color: '#333',
            maxWidth: 700,
            textAlign: { xs: 'center', md: 'left' }
          }}
        >
          Maximize dividend yield while maintaining portfolio diversity using entropy-based optimization. 
          Balance income generation with intelligent diversification.
        </Typography>
        <Typography variant="body2" sx={{ color: '#2e8b57', mb: 4, textAlign: { xs: 'center', md: 'left' } }}>
          Want risk-return optimization instead? Try our <Link href="/" style={{ color: '#0052cc', textDecoration: 'underline' }}>Portfolio Optimizer</Link>.
        </Typography>
      </Box>

      {/* Exchange Selection */}
      <Card sx={{ maxWidth: 900, mx: 'auto', mb: 3, p: 3 }}>
        <Typography variant="h6" component="h2" gutterBottom sx={{ fontWeight: 600 }}>
          Select Exchange
        </Typography>
        <FormControl fullWidth variant="outlined" sx={{ maxWidth: 600 }}>
          <InputLabel id="exchange-select-label">Choose Exchange</InputLabel>
          <Select
            labelId="exchange-select-label"
            value={selectedExchange || ''}
            onChange={(e) => setSelectedExchange(e.target.value as ExchangeEnum)}
            label="Choose Exchange"
          >
            <MenuItem value={ExchangeEnum.NSE}>NSE</MenuItem>
            <MenuItem value={ExchangeEnum.BSE}>BSE</MenuItem>
          </Select>
        </FormControl>
      </Card>

      {/* Stock Selection */}
      <Card sx={{ 
        maxWidth: 900, 
        mx: 'auto', 
        mb: 3, 
        p: 3,
        opacity: selectedExchange ? 1 : 0.5, 
        pointerEvents: selectedExchange ? 'auto' : 'none'
      }}>
        <Typography variant="h6" component="h2" gutterBottom sx={{ fontWeight: 600 }}>
          Select Dividend Stocks
        </Typography>
        <Alert severity="info" sx={{ mb: 2 }}>
          💡 Tip: Focus on high-dividend yield stocks like PSUs (ONGC, NTPC, Coal India), utilities, and oil companies.
        </Alert>
        <Autocomplete
          options={filteredOptions}
          getOptionLabel={(o) => `${o.ticker} - ${o.name} (${o.exchange})`}
          onChange={handleAddStock}
          inputValue={inputValue}
          onInputChange={(e, v) => setInputValue(v)}
          renderInput={(params) => (
            <TextField 
              {...params} 
              label="Search Dividend Stock" 
              variant="outlined"
              placeholder="Type to search dividend-paying stocks..."
            />
          )}
          sx={{ width: '100%', maxWidth: 600 }}
          openOnFocus={true}
          filterOptions={(x) => x}
          clearOnBlur={false}
          value={null}
        />
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 500 }}>
            Selected Stocks ({selectedStocks.length})
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            {selectedStocks.map((stock, idx) => (
              <Chip
                key={idx}
                label={`${stock.ticker} (${stock.exchange})`}
                onDelete={() => handleRemoveStock(stock)}
                sx={{ m: 0.5 }}
                color="primary"
                variant="outlined"
              />
            ))}
          </Stack>
        </Box>
      </Card>

      {/* Investment & Optimization Parameters */}
      <Card sx={{ 
        maxWidth: 900, 
        mx: 'auto', 
        mb: 3, 
        p: 3,
        opacity: selectedExchange ? 1 : 0.5, 
        pointerEvents: selectedExchange ? 'auto' : 'none'
      }}>
        <Typography variant="h6" component="h2" gutterBottom sx={{ fontWeight: 600 }}>
          Investment & Optimization Parameters
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Investment Budget (Optional)"
              type="number"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              variant="outlined"
              InputProps={{
                startAdornment: <InputAdornment position="start">₹</InputAdornment>,
                inputProps: { min: 0, step: 10000 }
              }}
              helperText="Enter amount to see share allocation. Leave empty for weights only."
            />
          </Grid>
          
          <Grid item xs={12}>
            <Typography gutterBottom>
              Entropy Weight (λ): {entropyWeight.toFixed(2)}
            </Typography>
            <Tooltip title="Higher values increase diversification, lower values focus more on yield">
              <Slider
                value={entropyWeight}
                onChange={(e, v) => setEntropyWeight(v as number)}
                min={0}
                max={0.5}
                step={0.01}
                marks={[
                  { value: 0, label: 'Pure Yield' },
                  { value: 0.05, label: '0.05 (Default)' },
                  { value: 0.1, label: '0.1' },
                  { value: 0.25, label: '0.25' },
                  { value: 0.5, label: 'Max Diversification' }
                ]}
                valueLabelDisplay="auto"
              />
            </Tooltip>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={useVolCap}
                  onChange={(e) => {
                    setUseVolCap(e.target.checked);
                    if (e.target.checked && !volCap) {
                      setVolCap(0.25); // Default 25% volatility cap
                    }
                  }}
                />
              }
              label="Apply Volatility Cap"
            />
            {useVolCap && (
              <TextField
                fullWidth
                label="Max Annual Volatility"
                type="number"
                value={volCap || ''}
                onChange={(e) => setVolCap(parseFloat(e.target.value) || null)}
                variant="outlined"
                sx={{ mt: 1 }}
                helperText="E.g., 0.25 = 25% annual volatility"
                inputProps={{ min: 0.05, max: 1, step: 0.05 }}
              />
            )}
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Switch
                  checked={useMedianTtm}
                  onChange={(e) => setUseMedianTtm(e.target.checked)}
                />
              }
              label="Use Median TTM Yield"
            />
            <Typography variant="caption" color="text.secondary" display="block">
              Stabilizes yield calculation by using median of trailing 90 days
            </Typography>
          </Grid>
        </Grid>
      </Card>

      {/* Action Buttons */}
      <Box sx={{ 
        maxWidth: 900, 
        mx: 'auto', 
        mb: 3,
        display: 'flex',
        gap: 2,
        justifyContent: 'flex-start'
      }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={!canSubmit || loading}
          sx={{
            px: 3,
            py: 1,
            fontWeight: 600,
            background: !canSubmit || loading ? 'gray' : 'linear-gradient(90deg, #2e8b57 30%, #1976d2 100%)',
          }}
        >
          {loading ? 'Optimizing...' : 'Optimize Portfolio'}
        </Button>
        <Button 
          variant="outlined" 
          color="secondary" 
          onClick={handleReset}
          sx={{ px: 3, py: 1, fontWeight: 600 }}
        >
          Reset
        </Button>
        {!canSubmit && (
          <Typography color="error" sx={{ alignSelf: 'center' }}>
            {submitError}
          </Typography>
        )}
      </Box>

      {/* Loading and Results */}
      {loading ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
          <CircularProgress />
          <Typography sx={{ mt: 2, fontSize: '1.1rem', fontWeight: 500 }}>
            Optimizing Dividend Portfolio
          </Typography>
          <Typography sx={{ mt: 1, fontSize: '0.9rem', color: '#666' }}>
            Fetching dividend data and calculating optimal weights...
          </Typography>
        </Box>
      ) : error ? (
        <Card sx={{ maxWidth: 900, mx: 'auto', backgroundColor: '#ffebee' }}>
          <CardContent>
            <Typography variant="h6" color="error" gutterBottom>
              Optimization Error
            </Typography>
            <Typography variant="body1" gutterBottom>
              {error.message}
            </Typography>
            {error.details && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" fontWeight="bold">
                  Details:
                </Typography>
                {typeof error.details === 'string' ? (
                  <Typography>{error.details}</Typography>
                ) : (
                  <pre style={{ whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: 200 }}>
                    {JSON.stringify(error.details, null, 2)}
                  </pre>
                )}
              </Box>
            )}
            <Button 
              variant="contained" 
              color="primary" 
              sx={{ mt: 2 }}
              onClick={() => setError(null)}
            >
              Dismiss
            </Button>
          </CardContent>
        </Card>
      ) : (
        optimizationResult && (
          <Card sx={{ maxWidth: 900, mx: 'auto', mb: 3 }}>
            <CardContent>
              <Typography variant="h5" sx={{ fontWeight: 600, mb: 3 }}>
                Optimization Results
              </Typography>
              
              {/* Portfolio Summary */}
              <Alert severity="success" sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Portfolio Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography><strong>Portfolio Yield:</strong> {formatPercentage(optimizationResult.portfolio_yield)}</Typography>
                    <Typography><strong>Entropy (Diversification):</strong> {optimizationResult.entropy.toFixed(3)}</Typography>
                    <Typography><strong>Effective N:</strong> {optimizationResult.effective_n.toFixed(2)} stocks</Typography>
                    <Typography><strong>Annual Volatility:</strong> {formatPercentage(Math.sqrt(optimizationResult.realized_variance))}</Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    {optimizationResult.budget && (
                      <>
                        <Typography><strong>Budget:</strong> {formatCurrency(optimizationResult.budget)}</Typography>
                        <Typography><strong>Amount Invested:</strong> {formatCurrency(optimizationResult.amount_invested || 0)}</Typography>
                        <Typography><strong>Residual Cash:</strong> {formatCurrency(optimizationResult.residual_cash || 0)}</Typography>
                        <Typography><strong>Deployment Rate:</strong> {formatPercentage(optimizationResult.deployment_rate || 0)}</Typography>
                        <Typography sx={{ color: '#2e8b57', fontWeight: 'bold' }}>
                          <strong>Annual Income:</strong> {formatCurrency(optimizationResult.annual_income || 0)}
                        </Typography>
                      </>
                    )}
                    <Typography><strong>Data Period:</strong> {new Date(optimizationResult.start_date).toLocaleDateString()} - {new Date(optimizationResult.end_date).toLocaleDateString()}</Typography>
                  </Grid>
                </Grid>
              </Alert>

              {/* Share Allocation Table (when budget is provided) */}
              {optimizationResult.shares && (
                <>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mt: 3 }}>
                    Share Allocation
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Stock</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Shares</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Price</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Investment</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Weight</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Yield</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Annual Income</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(optimizationResult.shares)
                        .filter(([_, shares]) => shares > 0)
                        .sort(([a], [b]) => (optimizationResult.weights[b] || 0) - (optimizationResult.weights[a] || 0))
                        .map(([ticker, shares]) => {
                          const price = optimizationResult.last_close[ticker] || 0;
                          const investment = shares * price;
                          const tickerYield = optimizationResult.per_ticker_yield[ticker] || 0;
                          const annualIncome = investment * tickerYield;
                          
                          return (
                            <TableRow key={ticker}>
                              <TableCell>{ticker}</TableCell>
                              <TableCell align="right">{shares.toLocaleString()}</TableCell>
                              <TableCell align="right">{formatCurrency(price)}</TableCell>
                              <TableCell align="right">{formatCurrency(investment)}</TableCell>
                              <TableCell align="right">{formatPercentage(optimizationResult.weights[ticker] || 0)}</TableCell>
                              <TableCell align="right">{formatPercentage(tickerYield)}</TableCell>
                              <TableCell align="right">{formatCurrency(annualIncome)}</TableCell>
                            </TableRow>
                          );
                        })}
                    </TableBody>
                  </Table>
                </>
              )}

              {/* Weights Table (always shown) */}
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600, mt: 3 }}>
                Optimal Weights
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Stock</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Weight</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Yield</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Last Price</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(optimizationResult.weights)
                    .filter(([_, weight]) => weight > 0.001)
                    .sort(([, a], [, b]) => b - a)
                    .map(([ticker, weight]) => (
                      <TableRow key={ticker}>
                        <TableCell>{ticker}</TableCell>
                        <TableCell align="right" sx={{ fontWeight: weight > 0.1 ? 'bold' : 'normal' }}>
                          {formatPercentage(weight)}
                        </TableCell>
                        <TableCell align="right">
                          {optimizationResult.per_ticker_yield[ticker] 
                            ? formatPercentage(optimizationResult.per_ticker_yield[ticker])
                            : 'N/A'}
                        </TableCell>
                        <TableCell align="right">
                          {optimizationResult.last_close[ticker] 
                            ? formatCurrency(optimizationResult.last_close[ticker])
                            : 'N/A'}
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>

              {/* Investment Calculator (if no budget provided) */}
              {!optimizationResult.budget && (
                <Alert severity="info" sx={{ mt: 3 }}>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    💡 Investment Calculator
                  </Typography>
                  <Typography variant="body2">
                    Enter a budget above to see exact share allocation. Example with ₹10,00,000:
                  </Typography>
                  <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                    <li>Expected Annual Dividend: {formatCurrency(1000000 * optimizationResult.portfolio_yield)}</li>
                    <li>Monthly Income: {formatCurrency((1000000 * optimizationResult.portfolio_yield) / 12)}</li>
                    <li>Quarterly Income: {formatCurrency((1000000 * optimizationResult.portfolio_yield) / 4)}</li>
                  </ul>
                </Alert>
              )}
            </CardContent>
          </Card>
        )
      )}
    </>
  );
};

export default DividendOptimizer;