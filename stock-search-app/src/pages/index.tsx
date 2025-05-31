import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { StockData, StockOption, PortfolioOptimizationResponse, OptimizationResult, APIError, ExchangeEnum, OptimizationMethod, CLAOptimizationMethod, BenchmarkName } from '../types';
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
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Legend,
  Tooltip as ChartTooltip,
} from 'chart.js';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
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
import GetApp from '@mui/icons-material/GetApp';
import InfoOutlined from '@mui/icons-material/InfoOutlined';
import Tooltip from '@mui/material/Tooltip';
import Box from '@mui/material/Box';

// Import TopNav component
import TopNav from '../components/TopNav';

// Register Chart.js components
ChartJS.register(
  LineElement, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  Legend, 
  ChartTooltip
);

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

// CLA sub-method options
const claSubOptions = [
  { label: 'Mean-Variance Optimization', value: 'MVO' },
  { label: 'Minimum Volatility', value: 'MinVol' },
  { label: 'Both', value: 'Both' },
];

// Algorithm options for selection
const algorithmOptions = [
  { label: 'Mean-Variance Optimization', value: 'MVO' },
  { label: 'Minimum Volatility', value: 'MinVol' },
  { label: 'Maximum Quadratic Utility', value: 'MaxQuadraticUtility' },
  { label: 'Equally Weighted', value: 'EquiWeighted' },
  { label: 'Critical Line Algorithm', value: 'CriticalLineAlgorithm' },
  { label: 'Hierarchical Risk Parity (HRP)', value: 'HRP' },
  { label: 'Hierarchical Equal Risk Contribution (HERC)', value: 'HERC' },
  { label: 'Nested Clustered Optimization (NCO)', value: 'NCO' },
  { label: 'Hierarchical Equal Risk Contribution 2 (HERC2)', value: 'HERC2' },
  { label: 'Minimum Conditional Value at Risk (CVaR)', value: 'MinCVaR' },
  { label: 'Minimum Conditional Drawdown at Risk (CDaR)', value: 'MinCDaR' },
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
  HERC: "Hierarchical Equal Risk Contribution (HERC)",
  NCO: "Nested Clustered Optimization (NCO)", 
  HERC2: "Hierarchical Equal Risk Contribution 2 (HERC2)",
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
  const [selectedBenchmark, setSelectedBenchmark] = useState<BenchmarkName>(BenchmarkName.nifty);
  const [selectedExchange, setSelectedExchange] = useState<ExchangeEnum | null>(null);

  // Default stocks to show initially - NSE
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
    { ticker: 'LT', name: 'Larsen & Toubro Ltd', exchange: 'NSE' },
    { ticker: 'AXISBANK', name: 'Axis Bank Ltd', exchange: 'NSE' },
    { ticker: 'ASIANPAINT', name: 'Asian Paints Ltd', exchange: 'NSE' },
    { ticker: 'MARUTI', name: 'Maruti Suzuki India Ltd', exchange: 'NSE' },
    { ticker: 'ULTRACEMCO', name: 'UltraTech Cement Ltd', exchange: 'NSE' },
    { ticker: 'TITAN', name: 'Titan Company Ltd', exchange: 'NSE' },
    { ticker: 'BAJFINANCE', name: 'Bajaj Finance Ltd', exchange: 'NSE' },
    { ticker: 'ONGC', name: 'Oil and Natural Gas Corporation Ltd', exchange: 'NSE' },
    { ticker: 'NESTLEIND', name: 'Nestle India Ltd', exchange: 'NSE' },
    { ticker: 'POWERGRID', name: 'Power Grid Corporation of India Ltd', exchange: 'NSE' },
    { ticker: 'TECHM', name: 'Tech Mahindra Ltd', exchange: 'NSE' },
    { ticker: 'SUNPHARMA', name: 'Sun Pharmaceutical Industries Ltd', exchange: 'NSE' },
    { ticker: 'NTPC', name: 'NTPC Ltd', exchange: 'NSE' },
    { ticker: 'WIPRO', name: 'Wipro Ltd', exchange: 'NSE' },
    { ticker: 'HCLTECH', name: 'HCL Technologies Ltd', exchange: 'NSE' },
    { ticker: 'COALINDIA', name: 'Coal India Ltd', exchange: 'NSE' },
    { ticker: 'JSWSTEEL', name: 'JSW Steel Ltd', exchange: 'NSE' },
    { ticker: 'TATAMOTORS', name: 'Tata Motors Ltd', exchange: 'NSE' },
    { ticker: 'ADANIPORTS', name: 'Adani Ports and Special Economic Zone Ltd', exchange: 'NSE' },
    { ticker: 'GRASIM', name: 'Grasim Industries Ltd', exchange: 'NSE' },
    { ticker: 'M&M', name: 'Mahindra & Mahindra Ltd', exchange: 'NSE' },
    { ticker: 'TATASTEEL', name: 'Tata Steel Ltd', exchange: 'NSE' },
    { ticker: 'BAJAJFINSV', name: 'Bajaj Finserv Ltd', exchange: 'NSE' },
    { ticker: 'EICHERMOT', name: 'Eicher Motors Ltd', exchange: 'NSE' },
    { ticker: 'SHREECEM', name: 'Shree Cement Ltd', exchange: 'NSE' },
    { ticker: 'HEROMOTOCO', name: 'Hero MotoCorp Ltd', exchange: 'NSE' },
    { ticker: 'DRREDDY', name: 'Dr. Reddy\'s Laboratories Ltd', exchange: 'NSE' },
    { ticker: 'BRITANNIA', name: 'Britannia Industries Ltd', exchange: 'NSE' },
    { ticker: 'INDUSINDBK', name: 'IndusInd Bank Ltd', exchange: 'NSE' },
    { ticker: 'CIPLA', name: 'Cipla Ltd', exchange: 'NSE' },
    { ticker: 'BPCL', name: 'Bharat Petroleum Corporation Ltd', exchange: 'NSE' },
    { ticker: 'HDFCLIFE', name: 'HDFC Life Insurance Company Ltd', exchange: 'NSE' },
    { ticker: 'UPL', name: 'UPL Ltd', exchange: 'NSE' },
    { ticker: 'IOC', name: 'Indian Oil Corporation Ltd', exchange: 'NSE' },
    { ticker: 'DIVISLAB', name: 'Divi\'s Laboratories Ltd', exchange: 'NSE' },
    { ticker: 'SBILIFE', name: 'SBI Life Insurance Company Ltd', exchange: 'NSE' },
    { ticker: 'HINDALCO', name: 'Hindalco Industries Ltd', exchange: 'NSE' },
    { ticker: 'TATACONSUM', name: 'Tata Consumer Products Ltd', exchange: 'NSE' },
    { ticker: 'APOLLOHOSP', name: 'Apollo Hospitals Enterprise Ltd', exchange: 'NSE' },
    { ticker: 'BAJAJ-AUTO', name: 'Bajaj Auto Ltd', exchange: 'NSE' },
    { ticker: 'ABB', name: 'ABB India Ltd', exchange: 'NSE' }
  ];

  // Default stocks to show initially - BSE
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
    { ticker: 'LT', name: 'Larsen & Toubro Ltd', exchange: 'BSE' },
    { ticker: 'AXISBANK', name: 'Axis Bank Ltd', exchange: 'BSE' },
    { ticker: 'ASIANPAINT', name: 'Asian Paints Ltd', exchange: 'BSE' },
    { ticker: 'MARUTI', name: 'Maruti Suzuki India Ltd', exchange: 'BSE' },
    { ticker: 'ULTRACEMCO', name: 'UltraTech Cement Ltd', exchange: 'BSE' },
    { ticker: 'TITAN', name: 'Titan Company Ltd', exchange: 'BSE' },
    { ticker: 'BAJFINANCE', name: 'Bajaj Finance Ltd', exchange: 'BSE' },
    { ticker: 'ONGC', name: 'Oil and Natural Gas Corporation Ltd', exchange: 'BSE' },
    { ticker: 'NESTLEIND', name: 'Nestle India Ltd', exchange: 'BSE' },
    { ticker: 'POWERGRID', name: 'Power Grid Corporation of India Ltd', exchange: 'BSE' },
    { ticker: 'TECHM', name: 'Tech Mahindra Ltd', exchange: 'BSE' },
    { ticker: 'SUNPHARMA', name: 'Sun Pharmaceutical Industries Ltd', exchange: 'BSE' },
    { ticker: 'NTPC', name: 'NTPC Ltd', exchange: 'BSE' },
    { ticker: 'WIPRO', name: 'Wipro Ltd', exchange: 'BSE' },
    { ticker: 'HCLTECH', name: 'HCL Technologies Ltd', exchange: 'BSE' },
    { ticker: 'COALINDIA', name: 'Coal India Ltd', exchange: 'BSE' },
    { ticker: 'JSWSTEEL', name: 'JSW Steel Ltd', exchange: 'BSE' },
    { ticker: 'TATAMOTORS', name: 'Tata Motors Ltd', exchange: 'BSE' },
    { ticker: 'ADANIPORTS', name: 'Adani Ports and Special Economic Zone Ltd', exchange: 'BSE' },
    { ticker: 'GRASIM', name: 'Grasim Industries Ltd', exchange: 'BSE' },
    { ticker: 'M&M', name: 'Mahindra & Mahindra Ltd', exchange: 'BSE' },
    { ticker: 'TATASTEEL', name: 'Tata Steel Ltd', exchange: 'BSE' },
    { ticker: 'BAJAJFINSV', name: 'Bajaj Finserv Ltd', exchange: 'BSE' },
    { ticker: 'EICHERMOT', name: 'Eicher Motors Ltd', exchange: 'BSE' },
    { ticker: 'SHREECEM', name: 'Shree Cement Ltd', exchange: 'BSE' },
    { ticker: 'HEROMOTOCO', name: 'Hero MotoCorp Ltd', exchange: 'BSE' },
    { ticker: 'DRREDDY', name: 'Dr. Reddy\'s Laboratories Ltd', exchange: 'BSE' },
    { ticker: 'BRITANNIA', name: 'Britannia Industries Ltd', exchange: 'BSE' },
    { ticker: 'INDUSINDBK', name: 'IndusInd Bank Ltd', exchange: 'BSE' },
    { ticker: 'CIPLA', name: 'Cipla Ltd', exchange: 'BSE' },
    { ticker: 'BPCL', name: 'Bharat Petroleum Corporation Ltd', exchange: 'BSE' },
    { ticker: 'HDFCLIFE', name: 'HDFC Life Insurance Company Ltd', exchange: 'BSE' },
    { ticker: 'UPL', name: 'UPL Ltd', exchange: 'BSE' },
    { ticker: 'IOC', name: 'Indian Oil Corporation Ltd', exchange: 'BSE' },
    { ticker: 'DIVISLAB', name: 'Divi\'s Laboratories Ltd', exchange: 'BSE' },
    { ticker: 'SBILIFE', name: 'SBI Life Insurance Company Ltd', exchange: 'BSE' },
    { ticker: 'HINDALCO', name: 'Hindalco Industries Ltd', exchange: 'BSE' },
    { ticker: 'TATACONSUM', name: 'Tata Consumer Products Ltd', exchange: 'BSE' },
    { ticker: 'APOLLOHOSP', name: 'Apollo Hospitals Enterprise Ltd', exchange: 'BSE' },
    { ticker: 'BAJAJ-AUTO', name: 'Bajaj Auto Ltd', exchange: 'BSE' },
    { ticker: 'ABB', name: 'ABB India Ltd', exchange: 'BSE' }
  ];

  // Get default stocks based on selected exchange
  const getDefaultStocks = () => {
    if (!selectedExchange) return [];
    return selectedExchange === ExchangeEnum.NSE ? defaultNSEStocks : defaultBSEStocks;
  };

  // Get available benchmarks based on selected exchange
  const getAvailableBenchmarks = () => {
    if (!selectedExchange) return [];
    return selectedExchange === ExchangeEnum.NSE 
      ? [BenchmarkName.nifty, BenchmarkName.bank_nifty]
      : [BenchmarkName.sensex];
  };

  // Check if any clustering algorithm is selected
  const hasClusteringAlgorithm = selectedAlgorithms.some(algo => 
    ['HERC', 'NCO', 'HERC2'].includes(algo.value)
  );
  
  // Require at least 3 stocks for clustering algorithms
  const needsMinimum3Stocks = hasClusteringAlgorithm && selectedStocks.length < 3;
  
  const canSubmit = selectedStocks.length >= 2 && selectedAlgorithms.length >= 1 && selectedExchange !== null && !needsMinimum3Stocks;
  
  let submitError = '';
  if (selectedStocks.length < 2 && selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 2 stocks and 1 optimization method.';
  } else if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks.';
  } else if (needsMinimum3Stocks) {
    submitError = 'Clustering algorithms (HERC, NCO, HERC2) require at least 3 stocks.';
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
        setFilteredOptions(getDefaultStocks());
        return;
      }
      const results = fuse.search(input);
      // Filter results based on selected exchange
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

  // Add new useEffect to handle exchange changes
  useEffect(() => {
    setFilteredOptions(getDefaultStocks());
  }, [selectedExchange]);

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
    
    // Extra check for clustering algorithms
    if (needsMinimum3Stocks) {
      // Show error message but don't proceed
      return;
    }
    
    setLoading(true);
    setError(null); // Clear any previous errors
    const dataToSend = {
      stocks: selectedStocks.map((s) => ({ ticker: s.ticker, exchange: s.exchange })),
      methods: selectedAlgorithms.map((a) => a.value),
      ...(selectedAlgorithms.some((a) => a.value === 'CriticalLineAlgorithm')
        ? { cla_method: selectedCLA.value }
        : {}),
      benchmark: selectedBenchmark
    };
    //TODO: change to backend url
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

  // Helper function to extract all unique years from the rolling betas data
  const getAllYears = (result: PortfolioOptimizationResponse): number[] => {
    const yearsSet = new Set<number>();
    
    // Loop through all methods that have results
    Object.entries(result.results).forEach(([methodKey, methodData]) => {
      // Check if the method has rolling_betas data
      if (methodData && methodData.rolling_betas) {
        // Add all years to the set (converting string keys to numbers)
        Object.keys(methodData.rolling_betas).forEach(yearStr => {
          yearsSet.add(parseInt(yearStr, 10));
        });
      }
    });
    
    // Convert set to array, sort numerically, and return
    return Array.from(yearsSet).sort((a, b) => a - b);
  };

  // Update prepareChartData to include the new MinCVaR key with its own color (cyan)
  const prepareChartData = (res: PortfolioOptimizationResponse) => {
    const labels = res.dates.map((d) => new Date(d).toLocaleDateString());
    const datasets = [];
    if (res.cumulative_returns.MVO?.length) {
      datasets.push({ label: 'Mean-Variance Optimization', data: res.cumulative_returns.MVO, borderColor: 'blue', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.MinVol?.length) {
      datasets.push({ label: 'Minimum Volatility', data: res.cumulative_returns.MinVol, borderColor: 'green', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.MaxQuadraticUtility?.length) {
      datasets.push({ label: 'Max Quadratic Utility', data: res.cumulative_returns.MaxQuadraticUtility, borderColor: 'purple', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.EquiWeighted?.length) {
      datasets.push({ label: 'Equally Weighted', data: res.cumulative_returns.EquiWeighted, borderColor: 'orange', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.CriticalLineAlgorithm_MVO?.length) {
      datasets.push({ label: 'CLA (MVO)', data: res.cumulative_returns.CriticalLineAlgorithm_MVO, borderColor: 'magenta', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.CriticalLineAlgorithm_MinVol?.length) {
      datasets.push({ label: 'CLA (MinVol)', data: res.cumulative_returns.CriticalLineAlgorithm_MinVol, borderColor: 'teal', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.HRP?.length) {
      datasets.push({ label: 'Hierarchical Risk Parity (HRP)', data: res.cumulative_returns.HRP, borderColor: 'brown', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.HERC?.length) {
      datasets.push({ label: 'Hierarchical Equal Risk Contribution (HERC)', data: res.cumulative_returns.HERC, borderColor: '#FF6B35', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.NCO?.length) {
      datasets.push({ label: 'Nested Clustered Optimization (NCO)', data: res.cumulative_returns.NCO, borderColor: '#2E8B57', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.HERC2?.length) {
      datasets.push({ label: 'Hierarchical Equal Risk Contribution 2 (HERC2)', data: res.cumulative_returns.HERC2, borderColor: '#8A2BE2', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.MinCVaR?.length) {
      datasets.push({ label: 'Minimum Conditional VaR (MCVar)', data: res.cumulative_returns.MinCVaR, borderColor: 'cyan', fill: false, pointRadius: 0, borderWidth: 1.5 });
    }
    if (res.cumulative_returns.MinCDaR?.length) {
      datasets.push({ label: 'Minimum Conditional Drawdown at Risk (CDaR)', data: res.cumulative_returns.MinCDaR, borderColor: '#800080', fill: false, pointRadius: 0, borderWidth: 1.5 }); // Dark purple
    }
    
    // Add benchmark returns with dashed line and distinct colors
    if (res.benchmark_returns?.length > 0) {
      const benchmark = res.benchmark_returns[0];
      const benchmarkName = benchmark.name.charAt(0).toUpperCase() + benchmark.name.slice(1).replace('_', ' ');
      
      // Assign different colors based on benchmark type
      let benchmarkColor;
      switch (benchmark.name) {
        case BenchmarkName.nifty:
          benchmarkColor = '#8B0000'; // Dark red
          break;
        case BenchmarkName.sensex:
          benchmarkColor = '#006400'; // Dark green
          break;
        case BenchmarkName.bank_nifty:
          benchmarkColor = '#4B0082'; // Indigo
          break;
        default:
          benchmarkColor = 'black';
      }
      
      datasets.push({ 
        label: benchmarkName, 
        data: benchmark.returns, 
        borderColor: benchmarkColor,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
        borderWidth: 1.5
      });
    }
    
    return { labels, datasets };
  };

  // Prepare chart data for Rolling Betas
  const prepareBetaChartData = (res: PortfolioOptimizationResponse) => {
    const years = getAllYears(res);
    const yearLabels = years.map(year => year.toString());
    
    const datasets = [
      // Reference line for market beta (1.0)
      {
        label: 'Market Beta (1.0)',
        data: Array(years.length).fill(1),
        borderColor: 'rgba(128, 128, 128, 0.7)',
        borderWidth: 1.5,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false
      }
    ];
    
    // Add datasets for each method that has rolling betas
    Object.entries(res.results)
      .filter(([_, methodData]) => methodData && methodData.rolling_betas)
      .forEach(([methodKey, methodData]) => {
        if (!methodData || !methodData.rolling_betas) return;
        
        datasets.push({
          label: algoDisplayNames[methodKey] || methodKey,
          data: years.map(year => methodData.rolling_betas?.[year] || null),
          borderColor: colors[methodKey as keyof typeof colors] || '#000000',
          fill: false,
          pointRadius: 4,
          borderWidth: 2,
          borderDash: [] // Adding required borderDash property
        });
      });
    
    return { labels: yearLabels, datasets };
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

  // Canvas cache to eliminate redundant html2canvas calls for the same elements across regenerations
  const canvasCache = useRef<Record<string, HTMLCanvasElement>>({});

  // Define common html2canvas options to be used consistently
  const commonCanvasOptions = {
    scale: 1.5,
    useCORS: true,
    logging: false,
    allowTaint: true,
    backgroundColor: null,
    imageTimeout: 0
  };

  // Helper function to render an element to canvas once and cache the result
  const renderOnce = async (
    el: HTMLElement,
    key: string,
    options: typeof commonCanvasOptions // Ensure consistent options are passed
  ): Promise<HTMLCanvasElement> => {
    if (!canvasCache.current[key]) {
      console.log(`Rendering ${key} to canvas for the first time with options:`, options);
      canvasCache.current[key] = await html2canvas(el, options);
    } else {
      console.log(`Using cached canvas for ${key}`);
    }
    return canvasCache.current[key];
  };

  // Clear canvas cache when optimization results change, so stale canvases aren't used
  useEffect(() => {
    if (optimizationResult) {
      canvasCache.current = {};
      console.log('Canvas cache cleared due to new optimization results or component mount');
    }
  }, [optimizationResult]);

  const generatePDF = async () => {
    if (!optimizationResult) return;

    const pdfCacheKey = selectedStocks.map(s => s.ticker).join('-') + '-' +
                     (optimizationResult.start_date || '') + '-' +
                     (optimizationResult.end_date || '');

    // If we have a cached PDF, use it immediately without showing loading overlay
    if (pdfCache[pdfCacheKey]) {
      const link = document.createElement('a');
      link.href = pdfCache[pdfCacheKey];
      link.download = `portfolio_optimization_${new Date().toISOString().split('T')[0]}.pdf`;
      link.click();
      console.log('Using cached PDF (full report)');
      return;
    }

    const startTime = performance.now();
    console.log('Starting PDF generation (full process)...');

    // Only show loading overlay for new report generation
    const loadingElement = document.createElement('div');
    loadingElement.setAttribute('id', 'pdf-loading-overlay');
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

    try {
      const now = new Date();
      const dateOptions: Intl.DateTimeFormatOptions = {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: true
      };
      const formattedDateTime = now.toLocaleString(undefined, dateOptions);

      const pdfDoc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const pageWidth = pdfDoc.internal.pageSize.getWidth();
      const pageHeight = pdfDoc.internal.pageSize.getHeight();
      const margin = 10;

      // ----------- 1. TITLE PAGE -----------
      pdfDoc.setFontSize(24);
      pdfDoc.text('Portfolio Optimization Report', pageWidth / 2, 40, { align: 'center' });
      pdfDoc.setFontSize(12);
      pdfDoc.text(`Generated on: ${formattedDateTime}`, pageWidth / 2, 50, { align: 'center' });
      pdfDoc.setFontSize(14);
      pdfDoc.text('Selected Stocks:', pageWidth / 2, 70, { align: 'center' });
      const stockList = selectedStocks.map(s => `${s.ticker} (${s.exchange})`).join(', ');
      pdfDoc.setFontSize(12);
      const splitStocks = pdfDoc.splitTextToSize(stockList, pageWidth - (margin * 2));
      pdfDoc.text(splitStocks, pageWidth / 2, 80, { align: 'center' });
      pdfDoc.text(`Time Period: ${formatDate(optimizationResult.start_date)} to ${formatDate(optimizationResult.end_date)}`, pageWidth / 2, 100, { align: 'center' });
      pdfDoc.text(`Risk-free Rate: ${(optimizationResult.risk_free_rate! * 100).toFixed(4)}%`, pageWidth / 2, 110, { align: 'center' });


      // ----------- 2. OPTIMIZATION METHOD RESULTS -----------
      const resultsContainer = document.getElementById('optimization-results');
      if (resultsContainer) {
        const methodCards = Array.from(resultsContainer.querySelectorAll('.method-card'));
        
        for (let i = 0; i < methodCards.length; i++) {
          const card = methodCards[i] as HTMLElement;
          const methodNameElement = card.querySelector('h5');
          const methodName = methodNameElement ? methodNameElement.textContent || 'Optimization Method' : `Method ${i + 1}`;
          
          // Generate a unique key for the card for caching purposes
          let cardCacheKey = `card-${methodName.replace(/\s+/g, '-')}`;
          const cardIdAttr = card.getAttribute('id') || card.getAttribute('data-method-key');
          if (cardIdAttr) cardCacheKey = `card-${cardIdAttr}`;
          else cardCacheKey = `card-idx-${i}` // Fallback key if no specific identifier found

          pdfDoc.addPage();
          pdfDoc.setFontSize(16);
          pdfDoc.text(methodName, pageWidth / 2, 20, { align: 'center' });

          try {
            const cardCanvas = await renderOnce(card, cardCacheKey, commonCanvasOptions);
            const cardImgData = cardCanvas.toDataURL('image/png');
            const cardImgWidth = pageWidth - (margin * 2);
            const cardImgHeight = (cardCanvas.height * cardImgWidth) / cardCanvas.width;

            if (cardImgHeight > pageHeight - 30) {
              const scaleFactor = (pageHeight - 30) / cardImgHeight;
              pdfDoc.addImage(cardImgData, 'PNG', margin, 30, cardImgWidth * scaleFactor, cardImgHeight * scaleFactor);
            } else {
              pdfDoc.addImage(cardImgData, 'PNG', margin, 30, cardImgWidth, cardImgHeight);
            }
          } catch (err) {
            console.error(`Failed to render card for ${methodName} (key: ${cardCacheKey}):`, err);
            pdfDoc.setFontSize(12);
            pdfDoc.text(`Unable to render optimization result for ${methodName}.`, margin, 40);
          }
        }
      }

      // ----------- 3. CUMULATIVE RETURNS CHART -----------
      const chartElement = document.querySelector('#cumulative-returns-chart canvas');
      if (chartElement) {
        pdfDoc.addPage();
        pdfDoc.setFontSize(16);
        pdfDoc.text('Cumulative Returns Over Time', pageWidth / 2, 20, { align: 'center' });
        try {
          const chartCanvas = await renderOnce(chartElement as HTMLElement, 'cumulative-returns-chart', commonCanvasOptions);
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
          const tableCanvas = await renderOnce(yearlyReturnsSection as HTMLElement, 'yearly-returns-table', commonCanvasOptions);
          const tableImgData = tableCanvas.toDataURL('image/png');
          const tableImgWidth = pageWidth - (margin * 2);
          const tableImgHeight = (tableCanvas.height * tableImgWidth) / tableCanvas.width;

          if (tableImgHeight > pageHeight - 30) {
            const scaleFactor = (pageHeight - 30) / tableImgHeight;
            pdfDoc.addImage(tableImgData, 'PNG', margin, 30, tableImgWidth * scaleFactor, tableImgHeight * scaleFactor);
          } else {
            pdfDoc.addImage(tableImgData, 'PNG', margin, 30, tableImgWidth, tableImgHeight);
          }
        } catch (err) {
          console.error('Error rendering yearly returns table:', err);
          pdfDoc.setFontSize(12);
          pdfDoc.text('Unable to render yearly returns table. The data may be too large.', margin, 40);
        }
      }

      // ----------- 5. COVARIANCE HEATMAP -----------
      if (optimizationResult.covariance_heatmap) {
        pdfDoc.addPage();
        pdfDoc.setFontSize(16);
        pdfDoc.text('Variance-Covariance Matrix', pageWidth / 2, 20, { align: 'center' });
        const covImgWidth = pageWidth - (margin * 2);
        const covImgHeight = covImgWidth * 0.8;
        try {
          console.log('Adding covariance heatmap directly from base64 data');
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

      const pdfOutput = pdfDoc.output('dataurlstring');
      setPdfCache(prev => ({
        ...prev,
        [pdfCacheKey]: pdfOutput
      }));

      const downloadLink = document.createElement('a');
      downloadLink.href = pdfOutput;
      downloadLink.download = `portfolio_optimization_${now.toISOString().split('T')[0]}.pdf`;
      downloadLink.click();

      const endTime = performance.now();
      console.log(`PDF generation completed in ${(endTime - startTime).toFixed(0)}ms`);

    } catch (error) {
      console.error('Error generating PDF (outer try/catch):', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      const loadingOverlay = document.getElementById('pdf-loading-overlay');
      if (loadingOverlay) {
        loadingOverlay.remove();
      }
    }
  };

  // Reset selections when exchange changes
  useEffect(() => {
    setSelectedStocks([]);
    setInputValue('');
    setSelectedAlgorithms([]);
    setSelectedCLA(claSubOptions[2]);
    setOptimizationResult(null);
    setError(null);
    
    // Set default benchmark based on exchange
    if (selectedExchange === ExchangeEnum.NSE) {
      setSelectedBenchmark(BenchmarkName.nifty);
    } else if (selectedExchange === ExchangeEnum.BSE) {
      setSelectedBenchmark(BenchmarkName.sensex);
    }
  }, [selectedExchange]);

  // Define common color scheme for algorithm visualization for consistency
  const colors = {
    MVO: 'rgba(75, 192, 192, 1)', 
    MinVol: 'rgba(153, 102, 255, 1)',
    MaxQuadraticUtility: 'rgba(255, 159, 64, 1)',
    EquiWeighted: 'rgba(255, 99, 132, 1)',
    'CriticalLineAlgorithm_MVO': 'rgba(54, 162, 235, 1)',
    'CriticalLineAlgorithm_MinVol': 'rgba(255, 206, 86, 1)',
    HRP: 'rgba(75, 192, 75, 1)',
    HERC: 'rgba(255, 107, 53, 1)',  // Orange-red
    NCO: 'rgba(46, 139, 87, 1)',    // Sea green
    HERC2: 'rgba(138, 43, 226, 1)', // Blue violet
    MinCVaR: 'rgba(255, 99, 255, 1)',
    MinCDaR: 'rgba(199, 99, 132, 1)'
  };

  return (
    <>
      <Head>
        <title>Indian Stock Portfolio Optimization Tool | Optimize NSE & BSE Stocks</title>
        <meta
          name="description"
          content="Optimize your Indian stock portfolio with AI-driven quantitative models. Supports NSE, BSE, Mean-Variance, CVaR, HERC, NCO, and more for robust, risk-managed investing."
        />
        <meta name="keywords" content="Indian stock portfolio optimization, NSE portfolio, BSE portfolio, mean-variance optimization, CVaR, HERC, NCO, quantitative investing India, risk management stocks" />
      </Head>
      
      <TopNav />
      
      <Box sx={{ maxWidth: 900, mx: 'auto', mt: 4, mb: 5, px: 2 }}>
        <Typography
          component="h1"
          variant="h4"
          sx={{ fontWeight: 700, mb: 2, color: '#222', textAlign: { xs: 'center', md: 'left' } }}
        >
          Indian Stock Portfolio Optimization Tool
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
          Optimize your Indian stock portfolio with AI-driven quantitative models—tailored for NSE and BSE. Run Mean-Variance, CVaR, HERC, NCO and more. India-specific benchmarks, rolling betas, and robust risk metrics empower you to build portfolios for the Indian market.
        </Typography>
        <Typography variant="body2" sx={{ color: '#2e8b57', mb: 4, textAlign: { xs: 'center', md: 'left' } }}>
          Learn more about <Link href="/docs" style={{ color: '#0052cc', textDecoration: 'underline' }}>portfolio optimization methods</Link> for Indian stocks.
        </Typography>
      </Box>

      {/* Exchange Selection Section */}
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

      {/* Stock Search Section */}
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
          Search and Select Stocks
        </Typography>
        <Autocomplete
          options={filteredOptions}
          getOptionLabel={(o) => `${o.ticker} - ${o.name} (${o.exchange})`}
          onChange={handleAddStock}
          inputValue={inputValue}
          onInputChange={(e, v) => setInputValue(v)}
          renderInput={(params) => (
            <TextField 
              {...params} 
              label="Search Stock" 
              variant="outlined"
              placeholder="Type to search stocks..."
              helperText="Start typing to search for stocks"
              InputProps={{
                ...params.InputProps,
                sx: {
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: inputValue ? 'inherit' : 'primary.main',
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                  },
                  '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'primary.main',
                    borderWidth: 2,
                  },
                },
              }}
            />
          )}
          style={{ width: '100%', maxWidth: 600 }}
          openOnFocus={true}
          filterOptions={(x) => x}
          clearOnBlur={false}
          value={null}
        />
        <div className="mt-4">
          <Typography variant="subtitle1" style={{ marginBottom: '8px', fontWeight: 500 }}>Selected Stocks</Typography>
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
          <Card style={{ marginTop: '1rem', backgroundColor: '#e8f4fd', maxWidth: 600, boxShadow: 'none', border: '1px solid #d0e8fa' }}>
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
            <FormControl fullWidth variant="outlined" style={{ marginTop: '16px', maxWidth: 600 }}>
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

      {/* Benchmark Selection Section */}
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
          Select Benchmark Index
        </Typography>
        <FormControl fullWidth variant="outlined" style={{ maxWidth: 600 }}>
          <InputLabel id="benchmark-select-label">Choose Benchmark</InputLabel>
          <Select
            labelId="benchmark-select-label"
            value={selectedBenchmark}
            onChange={(e) => setSelectedBenchmark(e.target.value as BenchmarkName)}
            label="Choose Benchmark"
          >
            {getAvailableBenchmarks().map((benchmark) => (
              <MenuItem key={benchmark} value={benchmark}>
                {benchmark === BenchmarkName.bank_nifty 
                  ? 'Bank Nifty'
                  : benchmark.charAt(0).toUpperCase() + benchmark.slice(1).replace('_', ' ')}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
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
            background: !canSubmit || loading ? 'gray' : 'linear-gradient(90deg, #2e8b57 30%, #0052cc 100%)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          }}
        >
          Submit
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

      {/* Loading Spinner and Optimization Results */}
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: '200px' }}>
          <CircularProgress />
          <div style={{ marginTop: '16px', fontSize: '1.1rem', fontWeight: 500 }}>
            Running Optimizations
          </div>
        </div>
      ) : error ? (
        <Card style={{ marginTop: '2rem', backgroundColor: '#ffebee', maxWidth: '900px', margin: '0 auto', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)' }}>
          <CardContent>
            <Typography variant="h6" color="error" gutterBottom>
              Optimization Error
            </Typography>
            <Typography variant="body1" gutterBottom>
              {(error as APIError).message}
            </Typography>
            {(error as APIError).details && (
              <div style={{ marginTop: '1rem' }}>
                <Typography variant="subtitle2" fontWeight="bold">
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
              style={{ marginTop: '1rem', padding: '6px 16px', fontSize: '0.9rem' }}
              onClick={() => setError(null)}
            >
              Dismiss
            </Button>
          </CardContent>
        </Card>
      ) : (
        optimizationResult && (
          <div id="optimization-results" className="results-container" style={{
            background: 'white', 
            borderRadius: '8px', 
            padding: '20px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
            border: '1px solid #f0f0f0',
            maxWidth: '900px',
            margin: '0 auto 24px auto'
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography 
                variant="h5" 
                sx={{ 
                  fontWeight: 600, 
                  color: '#1e293b'
                }}
              >
                Optimization Results
              </Typography>
              <Button
                variant="outlined"
                startIcon={<GetApp />}
                onClick={generatePDF}
                size="small"
              >
                Download PDF Report
              </Button>
            </Box>
            
            {/* Method Cards */}
            {Object.entries(optimizationResult.results).map(([methodKey, methodData], index) => {
              if (!methodData) return null;
              
              // Get display name for the method
              const displayName = algoDisplayNames[methodKey] || methodKey;
              
              return (
                <Card 
                  key={methodKey} 
                  id={`method-card-${methodKey}`}
                  data-method-key={methodKey}
                  className="method-card"
                  sx={{ 
                    mb: 4, 
                    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
                    border: '1px solid #f0f0f0'
                  }}
                >
                  <CardContent>
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, color: '#1e293b' }}>
                      {displayName}
                    </Typography>
                    
                    <Grid container spacing={3}>
                      {/* Weights Section - Move to below the metrics */}
                      
                      {/* Performance Metrics */}
                      <Grid item xs={12} md={4}>
                        <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                          Performance Metrics
                        </Typography>
                        
                        {/* Primary Metrics */}
                        <Table size="small">
                          <TableBody>
                            {methodData.performance && (
                              <>
                                <TableRow>
                                  <TableCell>Expected Return</TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.expected_return)}>
                                    {(methodData.performance.expected_return * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>Volatility</TableCell>
                                  <TableCell align="right">
                                    {(methodData.performance.volatility * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Sharpe Ratio
                                    <Tooltip title="Sharpe Ratio measures excess return per unit of total volatility. Calculated as (Annualized Return – Risk-Free Rate) / Annualized Volatility.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.sharpe.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Sortino Ratio
                                    <Tooltip title="Sortino Ratio measures excess return per unit of downside volatility. Calculated as (Annualized Return – Risk-Free Rate) / Annualized Downside Deviation.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.sortino.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>Maximum Drawdown</TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.max_drawdown)}>
                                    {(methodData.performance.max_drawdown * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>CAGR</TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.cagr)}>
                                    {(methodData.performance.cagr * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Portfolio Beta
                                    <Tooltip title="Beta is estimated through Ordinary Least Squares (OLS) regression using the Capital Asset Pricing Model (CAPM). The regression equation is: Ri - Rf = α + β(Rm - Rf) + ε, where Ri is the portfolio return, Rf is the risk-free rate, Rm is the market return, and ε is the error term. The coefficient β represents the portfolio's sensitivity to market movements.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.portfolio_beta.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Portfolio Alpha
                                    <Tooltip title="Alpha is the portfolio's excess return over what would be predicted by the Capital Asset Pricing Model (CAPM). It represents the portfolio manager's ability to generate returns through security selection rather than market movements. A positive alpha means the portfolio outperformed its benchmark on a risk-adjusted basis, while a negative alpha indicates underperformance.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.portfolio_alpha)}>
                                    {(methodData.performance.portfolio_alpha * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>R-Squared</TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.r_squared.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>VaR (95%)</TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.var_95)}>
                                    {(methodData.performance.var_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>CVaR (95%)</TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.cvar_95)}>
                                    {(methodData.performance.cvar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Blume Adjusted Beta
                                    <Tooltip title="Blume Adjusted Beta = 1 + 0.67·(β – 1), which shrinks beta toward 1 to account for historical mean reversion.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.blume_adjusted_beta.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Treynor Ratio
                                    <Tooltip title="The Treynor Ratio measures excess return per unit of market risk. It's calculated as (Portfolio Return - Risk-Free Rate) / Portfolio Beta. A higher Treynor Ratio indicates better risk-adjusted performance relative to market risk.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.treynor_ratio.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Information Ratio
                                    <Tooltip title="The Information Ratio measures the active return divided by the active risk (tracking error). It quantifies the excess return per unit of risk taken relative to the benchmark. Higher values indicate better risk-adjusted active returns.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.information_ratio.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Skewness
                                    <Tooltip title="Measures asymmetry of the returns distribution. Positive skew indicates a long right tail.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.skewness.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Kurtosis
                                    <Tooltip title="Measures tail heaviness. High kurtosis indicates fat tails (more extreme outcomes).">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.kurtosis.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Omega Ratio
                                    <Tooltip title="The Omega Ratio measures the relationship between the probability of gains and losses relative to a threshold (typically the risk-free rate). It's the ratio of the area above the threshold to the area below it in the returns distribution. Higher values indicate better risk-return characteristics.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.omega_ratio.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Calmar Ratio
                                    <Tooltip title="The Calmar Ratio is the ratio of annualized return to maximum drawdown. It measures the return per unit of downside risk, with higher values indicating better risk-adjusted performance.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.calmar_ratio.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    EVaR (95%)
                                    <Tooltip title="Entropic Value at Risk (EVaR) uses a Chernoff-bound formulation via the moment-generating function to bound tail losses more tightly than VaR/CVaR. It provides a more conservative risk estimate than traditional VaR measures.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {(methodData.performance.evar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    DaR (95%)
                                    <Tooltip title="Drawdown at Risk (DaR) represents the drawdown that won't be exceeded with 95% confidence. It's similar to VaR but for drawdowns instead of returns.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {(methodData.performance.dar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    CDaR (95%)
                                    <Tooltip title="Conditional Drawdown at Risk (CDaR) is the expected drawdown when the drawdown exceeds the DaR. It provides insight into the severity of extreme drawdowns.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {(methodData.performance.cdar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                              </>
                            )}
                          </TableBody>
                        </Table>
                        
                        {/* Advanced Beta and Cross-Moment Metrics Section */}
                        <TableRow>
                          <TableCell colSpan={2} style={{ backgroundColor: '#f5f5f5' }}>
                            <Typography variant="subtitle1" style={{ fontWeight: 'bold' }}>
                              Advanced Beta and Cross-Moment Metrics
                              <span style={{ 
                                backgroundColor: '#4caf50', 
                                color: 'white', 
                                padding: '2px 6px', 
                                borderRadius: '4px', 
                                fontSize: '0.7rem', 
                                marginLeft: '8px',
                                fontWeight: 'bold',
                                verticalAlign: 'middle'
                              }}>
                                NEW
                              </span>
                            </Typography>
                          </TableCell>
                        </TableRow>
                        
                        <Table size="small">
                          <TableBody>
                            <TableRow>
                              <TableCell>
                                <strong>Welch Beta</strong>
                                <Tooltip title="A robust alternative to traditional beta that uses winsorization to reduce the impact of extreme returns. It provides a more stable measure of market sensitivity that is less affected by outliers.">
                                  <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                </Tooltip>
                              </TableCell>
                              <TableCell align="right">
                                {methodData.performance.welch_beta !== undefined ? methodData.performance.welch_beta.toFixed(4) : 'N/A'}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>
                                <strong>Semi Beta</strong>
                                <Tooltip title="A downside beta that measures the portfolio's sensitivity to the benchmark only during down markets. Higher values indicate greater correlation with the benchmark during market declines.">
                                  <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                </Tooltip>
                              </TableCell>
                              <TableCell align="right">
                                {methodData.performance.semi_beta !== undefined ? methodData.performance.semi_beta.toFixed(4) : 'N/A'}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>
                                <strong>Coskewness</strong>
                                <Tooltip title="Measures the relationship between portfolio returns and squared market returns. Negative values suggest the portfolio tends to have negative returns when market volatility increases.">
                                  <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                </Tooltip>
                              </TableCell>
                              <TableCell align="right">
                                {methodData.performance.coskewness !== undefined ? methodData.performance.coskewness.toFixed(4) : 'N/A'}
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell>
                                <strong>Cokurtosis</strong>
                                <Tooltip title="Fourth cross-moment measuring the relationship between portfolio returns and extreme market returns. A high positive value indicates portfolio returns amplify extreme market movements.">
                                  <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                </Tooltip>
                              </TableCell>
                              <TableCell align="right">
                                {methodData.performance.cokurtosis !== undefined ? methodData.performance.cokurtosis.toFixed(4) : 'N/A'}
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                        
                        <Typography variant="subtitle1" style={{ marginTop: '1rem', fontWeight: 'bold' }}>
                          Weights
                        </Typography>
                        <Table size="small">
                          <TableBody>
                            {methodData.weights && Object.entries(methodData.weights).map(([ticker, weight]) => (
                              <TableRow key={ticker}>
                                <TableCell>{ticker}</TableCell>
                                <TableCell align="right">{(weight * 100).toFixed(2)}%</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Grid>
                      
                      {/* Visualizations - Returns Distribution and Drawdown */}
                      <Grid item xs={12} md={8}>
                        {methodData.returns_dist && (
                          <Box>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              Returns Distribution
                            </Typography>
                            <ImageComponent base64String={methodData.returns_dist} altText="Returns Distribution" />
                          </Box>
                        )}
                        
                        {methodData.max_drawdown_plot && (
                          <Box>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              Maximum Drawdown
                            </Typography>
                            <ImageComponent base64String={methodData.max_drawdown_plot} altText="Maximum Drawdown" />
                          </Box>
                        )}
                      </Grid>
                    </Grid>
                    
                    {/* Additional Portfolio Visualizations */}
                    <Grid container spacing={3} sx={{ mt: 2 }}>
                      {/* Efficient Frontier */}
                      {methodData.efficient_frontier_img && (
                        <Grid item xs={12} md={6}>
                          <Box>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              Efficient Frontier
                            </Typography>
                            <ImageComponent base64String={methodData.efficient_frontier_img} altText="Efficient Frontier" />
                          </Box>
                        </Grid>
                      )}
                      
                      {/* Weights Plot */}
                      {methodData.weights_plot && (
                        <Grid item xs={12} md={6}>
                          <Box>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              Portfolio Weights Visualization
                            </Typography>
                            <ImageComponent base64String={methodData.weights_plot} altText="Portfolio Weights" />
                          </Box>
                        </Grid>
                      )}
                      
                      {/* Dendrogram Plot */}
                      {methodData.dendrogram_plot && (
                        <Grid item xs={12} md={6}>
                          <Box>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              Hierarchical Clustering Dendrogram
                            </Typography>
                            <ImageComponent base64String={methodData.dendrogram_plot} altText="Hierarchical Clustering" />
                          </Box>
                        </Grid>
                      )}
                    </Grid>
                  </CardContent>
                </Card>
              );
            })}
            
            {/* Cumulative Returns Chart */}
            {optimizationResult.dates && optimizationResult.cumulative_returns && (
              <div id="cumulative-returns-chart" style={{ marginTop: '2rem' }}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.25rem', fontWeight: 600 }}>
                  Cumulative Returns Over Time
                </Typography>
                <div style={{ height: '400px' }}>
                  <Line 
                    data={prepareChartData(optimizationResult)} 
                    options={chartOptions} 
                    height={undefined}
                    width={undefined}
                  />
                </div>
              </div>
            )}
            
            {/* Rolling Betas Visualization */}
            {optimizationResult.results && Object.values(optimizationResult.results).some(method => method && method.rolling_betas) && (
              <Card style={{ marginTop: '2rem' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ fontSize: '1.25rem', fontWeight: 600 }}>
                    Rolling Betas by Year
                    <Tooltip title="Beta values calculated for each calendar year, showing how the portfolio's market sensitivity changes over time. Beta=1.0 indicates market-equivalent risk level.">
                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                    </Tooltip>
                  </Typography>
                  <div style={{ height: '400px' }}>
                    <Line
                      data={{
                        labels: getAllYears(optimizationResult).map(String),
                        datasets: [
                          // Reference line for market beta (1.0)
                          {
                            label: 'Market Beta (1.0)',
                            data: Array(getAllYears(optimizationResult).length).fill(1),
                            borderColor: 'rgba(128, 128, 128, 0.7)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            fill: false
                          },
                          // Actual beta values for each method
                          ...Object.entries(optimizationResult.results)
                            .filter(([_, methodData]) => methodData && methodData.rolling_betas)
                            .map(([methodKey, methodData]) => ({
                              label: algoDisplayNames[methodKey] || methodKey,
                              data: getAllYears(optimizationResult).map(year => 
                                methodData?.rolling_betas ? methodData.rolling_betas[year] || null : null
                              ),
                              borderColor: colors[methodKey as keyof typeof colors] || '#000000',
                              // Remove backgroundColor as it's not in the type definition
                              fill: false,
                              pointRadius: 3,
                              borderWidth: 2
                            }))
                        ]
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            beginAtZero: false,
                            grid: {
                              color: 'rgba(0, 0, 0, 0.1)'
                            },
                            title: {
                              display: true,
                              text: 'Beta Value',
                              font: {
                                weight: 'bold'
                              }
                            }
                          },
                          x: {
                            grid: {
                              color: 'rgba(0, 0, 0, 0.1)'
                            },
                            title: {
                              display: true,
                              text: 'Year',
                              font: {
                                weight: 'bold'
                              }
                            }
                          }
                        },
                        plugins: {
                          legend: {
                            display: true,
                            position: 'top'
                          },
                          tooltip: {
                            callbacks: {
                              label: function(context: any) {
                                const value = context.raw;
                                const formattedBeta = typeof value === 'number' ? value.toFixed(2) : 'N/A';
                                let riskLevel = '';
                                if (typeof value === 'number') {
                                  if (value > 1.2) {
                                    riskLevel = '🔴 High Risk';
                                  } else if (value > 0.8) {
                                    riskLevel = '🟡 Medium Risk';
                                  } else {
                                    riskLevel = '🟢 Low Risk';
                                  }
                                }
                                return [`${context.dataset.label}: ${formattedBeta}`, riskLevel];
                              }
                            }
                          }
                        }
                      }}
                    />
                  </div>
                </CardContent>
              </Card>
            )}
            
            {/* Yearly Returns Table */}
            {optimizationResult.stock_yearly_returns && Object.keys(optimizationResult.stock_yearly_returns).length > 0 && (
              <div className="yearly-returns-section" style={{ marginTop: '2rem' }}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.25rem', fontWeight: 600 }}>
                  Yearly Stock Returns
                </Typography>
                <div style={{ overflowX: 'auto' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Stock</TableCell>
                        {allYears.map(year => (
                          <TableCell key={year} align="right">{year}</TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(optimizationResult.stock_yearly_returns).map(([ticker, yearData]) => (
                        <TableRow key={ticker}>
                          <TableCell>{ticker}</TableCell>
                          {allYears.map(year => {
                            const returnValue = yearData[year];
                            return (
                              <TableCell 
                                key={`${ticker}-${year}`} 
                                align="right"
                                style={getReturnCellStyle(returnValue)}
                              >
                                {returnValue !== undefined ? `${(returnValue * 100).toFixed(2)}%` : 'N/A'}
                              </TableCell>
                            );
                          })}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            )}
            
            {/* Covariance Heatmap */}
            {optimizationResult.covariance_heatmap && (
              <div style={{ marginTop: '2rem' }}>
                <Typography variant="h6" gutterBottom sx={{ fontSize: '1.25rem', fontWeight: 600 }}>
                  Variance-Covariance Matrix
                </Typography>
                <ImageComponent base64String={optimizationResult.covariance_heatmap} altText="Covariance Heatmap" />
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
    </>
  );
};

export default HomePage;