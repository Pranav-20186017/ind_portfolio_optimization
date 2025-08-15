import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  StockData, 
  StockOption, 
  ExchangeEnum, 
  DividendOptimizationMethod,
  DividendOptimizationRequest,
  DividendOptimizationResponse,
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
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import Alert from '@mui/material/Alert';
import InputAdornment from '@mui/material/InputAdornment';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';

// Import TopNav component
import TopNav from '../components/TopNav';

// Method options for dividend optimization
const dividendMethodOptions = [
  { 
    label: 'AUTO (Recommended)', 
    value: DividendOptimizationMethod.AUTO,
    description: 'Automatically selects the best method based on portfolio complexity and constraints'
  },
  { 
    label: 'GREEDY', 
    value: DividendOptimizationMethod.GREEDY,
    description: 'Fast greedy allocation with round-repair optimization'
  },
  { 
    label: 'MILP', 
    value: DividendOptimizationMethod.MILP,
    description: 'Exact mixed-integer linear programming optimization (slower but optimal)'
  },
  { 
    label: 'AGGRESSIVE (Max Deployment)', 
    value: DividendOptimizationMethod.AGGRESSIVE,
    description: 'Prioritizes maximum capital deployment with relaxed constraints - deploys 95%+ of budget'
  },
];

const DividendOptimizer: React.FC = () => {
  const [stockData, setStockData] = useState<StockData>({});
  const [options, setOptions] = useState<StockOption[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedStocks, setSelectedStocks] = useState<StockOption[]>([]);
  const [filteredOptions, setFilteredOptions] = useState<StockOption[]>([]);
  const [selectedExchange, setSelectedExchange] = useState<ExchangeEnum | null>(null);
  const [selectedMethod, setSelectedMethod] = useState<DividendOptimizationMethod>(DividendOptimizationMethod.AUTO);
  const [budget, setBudget] = useState<string>('1000000'); // Default 10L
  const [maxPositionSize, setMaxPositionSize] = useState<string>('0.25'); // 25% max per position
  const [minPositions, setMinPositions] = useState<string>('4'); // Minimum 4 positions
  const [minYield, setMinYield] = useState<string>('0.005'); // 0.5% minimum yield
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<APIError | null>(null);
  const [optimizationResult, setOptimizationResult] = useState<DividendOptimizationResponse | null>(null);

  // Default stocks - same as main page
  const defaultNSEStocks: StockOption[] = [
    { ticker: 'TCS', name: 'Tata Consultancy Services Ltd', exchange: 'NSE' },
    { ticker: 'INFY', name: 'Infosys Ltd', exchange: 'NSE' },
    { ticker: 'RELIANCE', name: 'Reliance Industries Ltd', exchange: 'NSE' },
    { ticker: 'HDFCBANK', name: 'HDFC Bank Ltd', exchange: 'NSE' },
    { ticker: 'ICICIBANK', name: 'ICICI Bank Ltd', exchange: 'NSE' },
    { ticker: 'HINDUNILVR', name: 'Hindustan Unilever Ltd', exchange: 'NSE' },
    { ticker: 'ITC', name: 'ITC Ltd', exchange: 'NSE' },
    { ticker: 'SBIN', name: 'State Bank of India', exchange: 'NSE' },
    { ticker: 'BHARTIARTL', name: 'Bharti Airtel Ltd', exchange: 'NSE' },
    { ticker: 'KOTAKBANK', name: 'Kotak Mahindra Bank Ltd', exchange: 'NSE' },
    // Add more high-dividend yield stocks
    { ticker: 'COALINDIA', name: 'Coal India Ltd', exchange: 'NSE' },
    { ticker: 'ONGC', name: 'Oil and Natural Gas Corporation Ltd', exchange: 'NSE' },
    { ticker: 'NTPC', name: 'NTPC Ltd', exchange: 'NSE' },
    { ticker: 'POWERGRID', name: 'Power Grid Corporation of India Ltd', exchange: 'NSE' },
    { ticker: 'IOC', name: 'Indian Oil Corporation Ltd', exchange: 'NSE' },
  ];

  const defaultBSEStocks: StockOption[] = [
    { ticker: 'TCS', name: 'Tata Consultancy Services Ltd', exchange: 'BSE' },
    { ticker: 'INFY', name: 'Infosys Ltd', exchange: 'BSE' },
    { ticker: 'RELIANCE', name: 'Reliance Industries Ltd', exchange: 'BSE' },
    { ticker: 'HDFCBANK', name: 'HDFC Bank Ltd', exchange: 'BSE' },
    { ticker: 'ICICIBANK', name: 'ICICI Bank Ltd', exchange: 'BSE' },
    { ticker: 'HINDUNILVR', name: 'Hindustan Unilever Ltd', exchange: 'BSE' },
    { ticker: 'ITC', name: 'ITC Ltd', exchange: 'BSE' },
    { ticker: 'SBIN', name: 'State Bank of India', exchange: 'BSE' },
    { ticker: 'BHARTIARTL', name: 'Bharti Airtel Ltd', exchange: 'BSE' },
    { ticker: 'KOTAKBANK', name: 'Kotak Mahindra Bank Ltd', exchange: 'BSE' },
    { ticker: 'COALINDIA', name: 'Coal India Ltd', exchange: 'BSE' },
    { ticker: 'ONGC', name: 'Oil and Natural Gas Corporation Ltd', exchange: 'BSE' },
    { ticker: 'NTPC', name: 'NTPC Ltd', exchange: 'BSE' },
    { ticker: 'POWERGRID', name: 'Power Grid Corporation of India Ltd', exchange: 'BSE' },
    { ticker: 'IOC', name: 'Indian Oil Corporation Ltd', exchange: 'BSE' },
  ];

  // Get default stocks based on selected exchange
  const getDefaultStocks = () => {
    if (!selectedExchange) return [];
    return selectedExchange === ExchangeEnum.NSE ? defaultNSEStocks : defaultBSEStocks;
  };

  // Validation
  const canSubmit = selectedStocks.length >= 2 && 
                   selectedExchange !== null && 
                   budget && 
                   parseFloat(budget) >= 10000;  // Minimum ₹10,000

  let submitError = '';
  if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks.';
  } else if (!budget || parseFloat(budget) < 10000) {
    submitError = 'Minimum budget is ₹10,000.';
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
    
    const dataToSend: DividendOptimizationRequest = {
      stocks: selectedStocks.map((s) => ({ ticker: s.ticker, exchange: s.exchange as ExchangeEnum })),
      budget: parseFloat(budget),
      method: selectedMethod,
      max_position_size: parseFloat(maxPositionSize) || 0.25,
      min_positions: parseInt(minPositions) || 3,
      min_yield: parseFloat(minYield) || 0.005
    };
    
    try {
      const response = await axios.post('https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/dividend-optimize', dataToSend);
      console.log('Backend response:', response.data);
      const result = response.data as DividendOptimizationResponse;
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
    setMaxPositionSize('0.25');
    setMinPositions('4');
    setMinYield('0.005');
    setSelectedMethod(DividendOptimizationMethod.AUTO);
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

  // Format currency
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <>
      <Head>
        <title>Dividend Portfolio Optimizer | High-Yield Indian Stocks</title>
        <meta
          name="description"
          content="Optimize your dividend portfolio with Indian stocks. Maximize yield while managing risk with NSE & BSE stocks. AI-driven allocation for sustainable income."
        />
        <meta name="keywords" content="dividend portfolio optimization, high yield Indian stocks, NSE dividend stocks, BSE dividend stocks, income investing India" />
      </Head>
      
      <TopNav />
      
      <Box sx={{ maxWidth: 900, mx: 'auto', mt: 4, mb: 5, px: 2 }}>
        <Typography
          component="h1"
          variant="h4"
          sx={{ fontWeight: 700, mb: 2, color: '#222', textAlign: { xs: 'center', md: 'left' } }}
        >
          Dividend Portfolio Optimizer
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
          Build a high-yield dividend portfolio with Indian stocks. Our optimizer maximizes dividend income while managing risk through intelligent allocation across NSE and BSE stocks.
        </Typography>
        <Typography variant="body2" sx={{ color: '#2e8b57', mb: 4, textAlign: { xs: 'center', md: 'left' } }}>
          Want risk-return optimization instead? Try our <Link href="/" style={{ color: '#0052cc', textDecoration: 'underline' }}>Portfolio Optimizer</Link>.
        </Typography>
      </Box>

      {/* Exchange Selection */}
      <div className="mb-6" style={{ 
        background: 'white', 
        borderRadius: '8px', 
        padding: '20px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
        border: '1px solid #f0f0f0',
        maxWidth: '900px',
        margin: '0 auto 24px auto'
      }}>
        <Typography 
          variant="h6" 
          component="h2" 
          gutterBottom
          style={{ 
            fontWeight: 600, 
            fontSize: '1.25rem',
            color: '#1e293b',
            marginBottom: '16px'
          }}
        >
          Select Exchange
        </Typography>
        <FormControl fullWidth variant="outlined" style={{ maxWidth: 600 }}>
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
      </div>

      {/* Stock Selection */}
      <div className="mb-6" style={{ 
        opacity: selectedExchange ? 1 : 0.5, 
        pointerEvents: selectedExchange ? 'auto' : 'none',
        background: 'white', 
        borderRadius: '8px', 
        padding: '20px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
        border: '1px solid #f0f0f0',
        maxWidth: '900px',
        margin: '0 auto 24px auto'
      }}>
        <Typography 
          variant="h6" 
          component="h2" 
          gutterBottom
          style={{ 
            fontWeight: 600, 
            fontSize: '1.25rem',
            color: '#1e293b',
            marginBottom: '16px'
          }}
        >
          Search and Select Dividend Stocks
        </Typography>
        <Alert severity="info" sx={{ mb: 2 }}>
          Focus on high-dividend yield stocks like PSUs (ONGC, NTPC, Coal India), banks, and utilities for better income generation.
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
              helperText="Select stocks known for consistent dividend payments"
            />
          )}
          style={{ width: '100%', maxWidth: 600 }}
          openOnFocus={true}
          filterOptions={(x) => x}
          clearOnBlur={false}
          value={null}
        />
        <div className="mt-4">
          <Typography variant="subtitle1" style={{ marginBottom: '8px', fontWeight: 500 }}>
            Selected Stocks ({selectedStocks.length})
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            {selectedStocks.map((stock, idx) => (
              <Chip
                key={idx}
                label={`${stock.ticker} (${stock.exchange})`}
                onDelete={() => handleRemoveStock(stock)}
                className="m-1"
                color="primary"
                variant="outlined"
              />
            ))}
          </Stack>
        </div>
      </div>

      {/* Budget and Method Selection */}
      <div className="mb-6" style={{ 
        opacity: selectedExchange ? 1 : 0.5, 
        pointerEvents: selectedExchange ? 'auto' : 'none',
        background: 'white', 
        borderRadius: '8px', 
        padding: '20px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
        border: '1px solid #f0f0f0',
        maxWidth: '900px',
        margin: '0 auto 24px auto'
      }}>
        <Typography 
          variant="h6" 
          component="h2" 
          gutterBottom
          style={{ 
            fontWeight: 600, 
            fontSize: '1.25rem',
            color: '#1e293b',
            marginBottom: '16px'
          }}
        >
          Investment Parameters
        </Typography>
        
        <Alert severity="info" sx={{ mb: 2 }}>
          New approach: Focus on income generation and full capital deployment. Risk/volatility is secondary in dividend investing.
        </Alert>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Investment Budget"
              type="number"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              variant="outlined"
              InputProps={{
                startAdornment: <InputAdornment position="start">₹</InputAdornment>,
                inputProps: { min: 10000, step: 10000 }
              }}
              helperText="Total amount to invest (min ₹10,000)"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Optimization Method</InputLabel>
              <Select
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value as DividendOptimizationMethod)}
                label="Optimization Method"
              >
                {dividendMethodOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    <Tooltip title={option.description} placement="right">
                      <span>{option.label}</span>
                    </Tooltip>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Max Position Size"
              type="number"
              value={maxPositionSize}
              onChange={(e) => setMaxPositionSize(e.target.value)}
              variant="outlined"
              InputProps={{
                inputProps: { min: 0.05, max: 0.5, step: 0.05 }
              }}
              helperText="Max % per stock (0.25 = 25%)"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Min Positions"
              type="number"
              value={minPositions}
              onChange={(e) => setMinPositions(e.target.value)}
              variant="outlined"
              InputProps={{
                inputProps: { min: 2, max: 20 }
              }}
              helperText="Minimum stocks for diversification"
            />
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Min Yield"
              type="number"
              value={minYield}
              onChange={(e) => setMinYield(e.target.value)}
              variant="outlined"
              InputProps={{
                inputProps: { min: 0, max: 0.1, step: 0.001 }
              }}
              helperText="Min acceptable yield (0.005 = 0.5%)"
            />
          </Grid>
        </Grid>
      </div>



      {/* Action Buttons */}
      <div className="mb-8" style={{ 
        background: 'white', 
        borderRadius: '8px', 
        padding: '20px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
        border: '1px solid #f0f0f0',
        maxWidth: '900px',
        margin: '0 auto 24px auto',
        display: 'flex',
        justifyContent: 'flex-start',
        gap: '16px'
      }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSubmit}
          disabled={!canSubmit || loading}
          style={{
            padding: '8px 24px',
            fontWeight: 600,
            fontSize: '0.95rem',
            borderRadius: '6px',
            background: !canSubmit || loading ? 'gray' : 'linear-gradient(90deg, #2e8b57 30%, #1976d2 100%)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          }}
        >
          {loading ? 'Optimizing...' : 'Optimize Portfolio'}
        </Button>
        <Button 
          variant="outlined" 
          color="secondary" 
          onClick={handleReset}
          style={{
            padding: '8px 24px',
            fontWeight: 600,
            fontSize: '0.95rem',
            borderRadius: '6px',
          }}
        >
          Reset
        </Button>
        {!canSubmit && (
          <Typography color="error" style={{ marginTop: '0.5rem' }}>
            {submitError}
          </Typography>
        )}
      </div>

      {/* Loading and Results */}
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: '200px' }}>
          <CircularProgress />
          <div style={{ marginTop: '16px', fontSize: '1.1rem', fontWeight: 500 }}>
            Optimizing Dividend Portfolio
          </div>
          <div style={{ marginTop: '8px', fontSize: '0.9rem', color: '#666' }}>
            This may take a few moments...
          </div>
        </div>
      ) : error ? (
        <Card style={{ marginTop: '2rem', backgroundColor: '#ffebee', maxWidth: '900px', margin: '0 auto' }}>
          <CardContent>
            <Typography variant="h6" color="error" gutterBottom>
              Optimization Error
            </Typography>
            <Typography variant="body1" gutterBottom>
              {error.message}
            </Typography>
            {error.details && (
              <div style={{ marginTop: '1rem' }}>
                <Typography variant="subtitle2" fontWeight="bold">
                  Details:
                </Typography>
                {Array.isArray(error.details) ? (
                  <ul>
                    {error.details.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                ) : typeof error.details === 'object' ? (
                  <pre style={{ whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: '200px' }}>
                    {JSON.stringify(error.details, null, 2)}
                  </pre>
                ) : (
                  <Typography>{String(error.details)}</Typography>
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
          <div style={{
            background: 'white', 
            borderRadius: '8px', 
            padding: '20px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
            border: '1px solid #f0f0f0',
            maxWidth: '900px',
            margin: '0 auto 24px auto'
          }}>
            <Typography 
              variant="h5" 
              sx={{ 
                fontWeight: 600, 
                color: '#1e293b',
                mb: 3
              }}
            >
              Dividend Portfolio Results
            </Typography>
            
            {/* Portfolio Summary */}
            <Card sx={{ mb: 3, backgroundColor: '#f8fffe', border: '1px solid #d1fae5' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                  Portfolio Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography><strong>Total Budget:</strong> {formatCurrency(optimizationResult.total_budget)}</Typography>
                    <Typography><strong>Amount Invested:</strong> {formatCurrency(optimizationResult.amount_invested)}</Typography>
                    <Typography><strong>Residual Cash:</strong> {formatCurrency(optimizationResult.residual_cash)}</Typography>
                    <Typography sx={{ 
                      color: optimizationResult.deployment_rate >= 0.99 ? 'green' : 
                             optimizationResult.deployment_rate >= 0.95 ? 'orange' : 'red',
                      fontWeight: 'bold'
                    }}>
                      <strong>🎯 Deployment Rate:</strong> {formatPercentage(optimizationResult.deployment_rate)}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography sx={{ fontWeight: 'bold', color: '#2e8b57' }}>
                      <strong>💰 Annual Income:</strong> {formatCurrency(optimizationResult.annual_income)}
                    </Typography>
                    <Typography><strong>Portfolio Yield:</strong> {formatPercentage(optimizationResult.portfolio_yield)}</Typography>
                    <Typography><strong>Yield on Invested:</strong> {formatPercentage(optimizationResult.yield_on_invested)}</Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography><strong>Method Used:</strong> {optimizationResult.allocation_method}</Typography>
                    <Typography><strong>Number of Positions:</strong> {optimizationResult.allocations.length}</Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Stock Allocations */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Stock Allocations
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 'bold' }}>Stock</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Shares</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Price</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Value</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        <Tooltip title="Weight as % of total budget">
                          <span>Weight</span>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        <Tooltip title="Weight as % of invested amount">
                          <span>Weight (Inv)</span>
                        </Tooltip>
                      </TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Yield</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Annual Income</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {optimizationResult.allocations
                      .sort((a, b) => b.value - a.value)
                      .map((allocation) => (
                        <TableRow key={allocation.symbol}>
                          <TableCell>{allocation.symbol}</TableCell>
                          <TableCell align="right">{allocation.shares.toLocaleString()}</TableCell>
                          <TableCell align="right">{formatCurrency(allocation.price)}</TableCell>
                          <TableCell align="right">{formatCurrency(allocation.value)}</TableCell>
                          <TableCell align="right" sx={{ fontWeight: allocation.weight > 0.1 ? 'bold' : 'normal' }}>
                            {formatPercentage(allocation.weight)}
                          </TableCell>
                          <TableCell align="right" sx={{ fontWeight: (allocation.weight_on_invested || allocation.weight) > 0.1 ? 'bold' : 'normal', color: '#2e8b57' }}>
                            {formatPercentage(allocation.weight_on_invested || allocation.weight)}
                          </TableCell>
                          <TableCell align="right">{formatPercentage(allocation.forward_yield)}</TableCell>
                          <TableCell align="right">{formatCurrency(allocation.annual_income)}</TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Dividend Data */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  Dividend Information
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ fontWeight: 'bold' }}>Stock</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Price</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Forward Dividend</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>Forward Yield</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 'bold' }}>Source</TableCell>
                      <TableCell align="center" sx={{ fontWeight: 'bold' }}>Confidence</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {optimizationResult.dividend_data.map((stock) => (
                      <TableRow key={stock.symbol}>
                        <TableCell>{stock.symbol}</TableCell>
                        <TableCell align="right">{formatCurrency(stock.price)}</TableCell>
                        <TableCell align="right">{formatCurrency(stock.forward_dividend)}</TableCell>
                        <TableCell align="right">{formatPercentage(stock.forward_yield)}</TableCell>
                        <TableCell align="center">
                          <Chip 
                            size="small" 
                            label={stock.dividend_source} 
                            color={stock.dividend_source === 'info' ? 'primary' : 'default'}
                          />
                        </TableCell>
                        <TableCell align="center">
                          {stock.confidence && (
                            <Chip 
                              size="small" 
                              label={stock.confidence} 
                              color={
                                stock.confidence === 'high' ? 'success' : 
                                stock.confidence === 'medium' ? 'warning' : 'error'
                              }
                            />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Sector Allocations */}
            {optimizationResult.sector_allocations && Object.keys(optimizationResult.sector_allocations).length > 0 && (
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                    Sector Allocation
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell sx={{ fontWeight: 'bold' }}>Sector</TableCell>
                        <TableCell align="right" sx={{ fontWeight: 'bold' }}>Allocation</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(optimizationResult.sector_allocations)
                        .sort(([,a], [,b]) => b - a)
                        .map(([sector, allocation]) => (
                          <TableRow key={sector}>
                            <TableCell>{sector}</TableCell>
                            <TableCell align="right">{formatPercentage(allocation)}</TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}

            {/* Optimization Summary */}
            {optimizationResult.optimization_summary && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6">Optimization Summary</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <pre style={{ 
                    whiteSpace: 'pre-wrap', 
                    fontSize: '0.85rem',
                    backgroundColor: '#f5f5f5',
                    padding: '16px',
                    borderRadius: '4px',
                    overflow: 'auto'
                  }}>
                    {JSON.stringify(optimizationResult.optimization_summary, null, 2)}
                  </pre>
                </AccordionDetails>
              </Accordion>
            )}
          </div>
        )
      )}

      {/* Helpful Tips */}
      <Card style={{ 
        marginTop: '2rem', 
        backgroundColor: '#f8f9ff', 
        maxWidth: '900px', 
        margin: '2rem auto',
        border: '1px solid #e3f2fd'
      }}>
        <CardContent>
          <Typography variant="h6" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
            💡 New Approach: Practitioner-Focused Dividend Investing
          </Typography>
          <ul style={{ margin: 0, paddingLeft: '20px' }}>
            <li><strong>Full Capital Deployment:</strong> Our optimizer targets 99%+ deployment rate - no cash sitting idle.</li>
            <li><strong>Income First:</strong> Maximizes annual dividend income, not theoretical risk metrics.</li>
            <li><strong>High-Yield Stocks:</strong> PSUs like ONGC, Coal India, NTPC, Power Grid often yield 5-8%+.</li>
            <li><strong>Simple Diversification:</strong> Use min positions (e.g., 4-6 stocks) for practical portfolios.</li>
            <li><strong>Position Sizing:</strong> Max 25-30% per stock prevents over-concentration.</li>
            <li><strong>Tax Note:</strong> Dividends are taxed at slab rates - factor this into your strategy.</li>
            <li><strong>Reinvestment Strategy:</strong> Use dividends to buy more shares quarterly for compounding.</li>
          </ul>
        </CardContent>
      </Card>
    </>
  );
};

export default DividendOptimizer;
