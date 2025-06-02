import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { StockData, StockOption, PortfolioOptimizationResponse, OptimizationResult, APIError, ExchangeEnum, OptimizationMethod, CLAOptimizationMethod, BenchmarkName, TechnicalIndicator, TechnicalIndicatorType } from '../types';
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
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CheckCircleOutline from '@mui/icons-material/CheckCircleOutline';

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
  { label: 'Equal Weighting', value: 'EquiWeighted' },
  { label: 'Critical Line Algorithm', value: 'CriticalLineAlgorithm' },
  { label: 'Hierarchical Risk Parity', value: 'HRP' },
  { label: 'Hierarchical Equal Risk Contribution', value: 'HERC' },
  { label: 'Hierarchical Equal Risk Contribution 2', value: 'HERC2' },
  { label: 'Nested Clustered Optimization', value: 'NCO' },
  { label: 'Minimize Conditional Value-at-Risk', value: 'MinCVaR' },
  { label: 'Minimize Conditional Drawdown-at-Risk', value: 'MinCDaR' },
  { label: 'Technical Indicator Optimization', value: 'TECHNICAL' },
];

// Method Display Names
const methodDisplayNames: Record<string, string> = {
  MVO: "Mean-Variance Optimization",
  MinVol: "Minimum Volatility",
  MaxQuadraticUtility: "Maximum Quadratic Utility",
  EquiWeighted: "Equal Weighting",
  CriticalLineAlgorithm_MVO: "Critical Line Alg. (MVO)",
  CriticalLineAlgorithm_MinVol: "Critical Line Alg. (MinVol)",
  HRP: "Hierarchical Risk Parity",
  MinCVaR: "Minimum Conditional Value-at-Risk",
  MinCDaR: "Minimum Conditional Drawdown-at-Risk",
  HERC: "Hierarchical Equal Risk Contribution",
  NCO: "Nested Clustered Optimization",
  HERC2: "Hierarchical Equal Risk Contribution 2",
  TECHNICAL: "Technical Indicator Optimization"
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
  
  // Technical indicators state
  const [selectedIndicators, setSelectedIndicators] = useState<TechnicalIndicator[]>([]);
  const [selectedIndicatorName, setSelectedIndicatorName] = useState<TechnicalIndicatorType | null>(null);
  const [selectedIndicatorWindow, setSelectedIndicatorWindow] = useState('');
  const [selectedIndicatorMult, setSelectedIndicatorMult] = useState('');
  
  // Predefined window options for each indicator type
  const indicatorWindowOptions: Record<TechnicalIndicatorType, number[]> = {
    [TechnicalIndicatorType.SMA]: [5, 10, 20, 50, 100, 200],
    [TechnicalIndicatorType.EMA]: [5, 10, 20, 50, 100, 200],
    [TechnicalIndicatorType.WMA]: [5, 10, 20, 50, 100],
    [TechnicalIndicatorType.RSI]: [7, 14, 21, 28],
    [TechnicalIndicatorType.WILLR]: [7, 14, 21, 28],
    [TechnicalIndicatorType.CCI]: [7, 14, 21, 28],
    [TechnicalIndicatorType.ROC]: [5, 10, 15, 20],
    [TechnicalIndicatorType.ATR]: [7, 14, 21, 28],
    [TechnicalIndicatorType.SUPERTREND]: [7, 10, 14, 21],
    [TechnicalIndicatorType.BBANDS]: [10, 20, 30, 40],
    [TechnicalIndicatorType.OBV]: [1, 2, 3, 5, 10],
    [TechnicalIndicatorType.AD]: [1, 2, 3, 5, 10]
  };
  
  // Update window size when indicator type changes
  useEffect(() => {
    if (selectedIndicatorName && indicatorWindowOptions[selectedIndicatorName]?.length > 0) {
      setSelectedIndicatorWindow(indicatorWindowOptions[selectedIndicatorName][0].toString());
    }
  }, [selectedIndicatorName]);
  
  // Check if technical optimization is selected
  const isTechnicalOptimizationSelected = selectedAlgorithms.some(algo => 
    algo.value === 'TECHNICAL'
  );
  
  // Function to add a technical indicator
  const handleAddIndicator = () => {
    if (selectedIndicatorName && selectedIndicatorWindow) {
      const windowValue = parseInt(selectedIndicatorWindow);
      if (isNaN(windowValue)) {
        return; // Don't add indicator if window isn't a valid number
      }
      
      const newIndicator: TechnicalIndicator = {
        name: selectedIndicatorName as TechnicalIndicatorType,
        window: windowValue,
        // Include mult only for SUPERTREND indicator
        ...(selectedIndicatorName === TechnicalIndicatorType.SUPERTREND && selectedIndicatorMult 
          ? { mult: parseFloat(selectedIndicatorMult) } 
          : {})
      };
      
      setSelectedIndicators([...selectedIndicators, newIndicator]);
      // Reset input values
      setSelectedIndicatorName(null);
      setSelectedIndicatorWindow('');
      setSelectedIndicatorMult('');
    }
  };
  
  // Function to remove a technical indicator
  const handleRemoveIndicator = (indexToRemove: number) => {
    setSelectedIndicators(selectedIndicators.filter((_, index) => index !== indexToRemove));
  };

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
  
  // Require at least 2 technical indicators for technical optimization
  const needsMinimum2Indicators = isTechnicalOptimizationSelected && selectedIndicators.length < 2;
  
  const canSubmit = selectedStocks.length >= 2 && 
                   selectedAlgorithms.length >= 1 && 
                   selectedExchange !== null && 
                   !needsMinimum3Stocks &&
                   !(isTechnicalOptimizationSelected && needsMinimum2Indicators);
  
  let submitError = '';
  if (selectedStocks.length < 2 && selectedAlgorithms.length < 1) {
    submitError = 'Please select at least 2 stocks and 1 optimization method.';
  } else if (selectedStocks.length < 2) {
    submitError = 'Please select at least 2 stocks.';
  } else if (needsMinimum3Stocks) {
    submitError = 'Clustering algorithms (HERC, NCO, HERC2) require at least 3 stocks.';
  } else if (needsMinimum2Indicators) {
    submitError = 'Technical Indicator Optimization requires at least 2 indicators.';
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
    
    // Extra check for technical indicators
    if (isTechnicalOptimizationSelected && needsMinimum2Indicators) {
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
      benchmark: selectedBenchmark,
      ...(isTechnicalOptimizationSelected 
        ? { indicators: selectedIndicators }
        : {})
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

  // Check if the result is from technical optimization
  const isTechnicalOptimizationResult = optimizationResult?.is_technical_only;

  // Reset technical indicators when clearing results
  const handleReset = () => {
    setSelectedStocks([]);
    setInputValue('');
    setSelectedAlgorithms([]);
    setSelectedCLA(claSubOptions[2]);
    setOptimizationResult(null);
    setError(null); // Clear any errors on reset
    setSelectedIndicators([]); // Clear technical indicators
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

  // Add a helper to safely get the method display name
  const getMethodDisplayName = (methodKey: string): string => {
    return methodDisplayNames[methodKey] || methodKey;
  };

  // Then update the prepareBetaChartData function
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
    
    // Add datasets for each method that has rolling betas, filtering out TECHNICAL
    Object.entries(res.results)
      .filter(([methodKey, methodData]) => methodKey !== "TECHNICAL" && methodData && methodData.rolling_betas)
      .forEach(([methodKey, methodData]) => {
        if (!methodData || !methodData.rolling_betas) return;
        
        datasets.push({
          label: getMethodDisplayName(methodKey),
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

    // Special handling for technical optimization
    const isTechnicalReport = isTechnicalOptimizationResult;
    
    // For technical optimization, show special message
    if (isTechnicalReport) {
      alert("Note: PDF report for Technical Indicator Optimization will only include optimization results and weights.");
    }

    const pdfCacheKey = selectedStocks.map(s => s.ticker).join('-') + '-' +
                     (optimizationResult.start_date || '') + '-' +
                     (optimizationResult.end_date || '') +
                     (isTechnicalReport ? '-technical' : '');

    // If we have a cached PDF, use it immediately without showing loading overlay
    if (pdfCache[pdfCacheKey]) {
      const link = document.createElement('a');
      link.href = pdfCache[pdfCacheKey];
      link.download = `portfolio_optimization_${new Date().toISOString().split('T')[0]}${isTechnicalReport ? '_technical' : ''}.pdf`;
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
        // Fix: Instead of looking for a class that doesn't exist, get all Cards directly
        // Get all card elements in the results container - these are the method cards
        const methodCards = Array.from(resultsContainer.querySelectorAll('div[class*="MuiCard-root"]')).filter(
          // Exclude any cards that are part of other components like the date info box
          card => {
            // Only include cards that have a Typography with a method name (portfolio heading)
            const hasMethodName = card.querySelector('h5, [class*="MuiTypography-h5"]');
            // Exclude cards that are clearly not method cards (like info boxes)
            const isInfoCard = card.querySelector('[class*="infoCard"], [class*="Information"]');
            return hasMethodName && !isInfoCard;
          }
        );
        
        console.log(`Found ${methodCards.length} method cards for PDF generation`);
        
        for (let i = 0; i < methodCards.length; i++) {
          const card = methodCards[i] as HTMLElement;
          
          // Extract method name from the card's heading
          const methodNameElement = card.querySelector('h5, [class*="MuiTypography-h5"]');
          const methodName = methodNameElement ? methodNameElement.textContent || 'Optimization Method' : `Method ${i + 1}`;
          
          // Generate a unique key for the card for caching purposes
          let cardCacheKey = `method-card-${i}-${methodName.replace(/\s+/g, '-')}`;
          
          pdfDoc.addPage();
          pdfDoc.setFontSize(16);
          pdfDoc.text(methodName, pageWidth / 2, 20, { align: 'center' });

          try {
            // Increase scale for better quality
            const cardCanvas = await renderOnce(card, cardCacheKey, {
              ...commonCanvasOptions,
              scale: 2.0 // Higher scale for better quality
            });
            
            const cardImgData = cardCanvas.toDataURL('image/png');
            const cardImgWidth = pageWidth - (margin * 2);
            const cardImgHeight = (cardCanvas.height * cardImgWidth) / cardCanvas.width;

            // If the image is too tall for a single page, scale it down
            if (cardImgHeight > (pageHeight - 30)) {
              const scaleFactor = (pageHeight - 30) / cardImgHeight;
              pdfDoc.addImage(
                cardImgData, 
                'PNG', 
                margin, 
                30, 
                cardImgWidth * scaleFactor, 
                cardImgHeight * scaleFactor
              );
              console.log(`Rendered card ${i+1}/${methodCards.length} (${methodName}) scaled down to fit page`);
            } else {
              pdfDoc.addImage(cardImgData, 'PNG', margin, 30, cardImgWidth, cardImgHeight);
              console.log(`Rendered card ${i+1}/${methodCards.length} (${methodName}) at full size`);
            }
          } catch (err) {
            console.error(`Failed to render card for ${methodName} (key: ${cardCacheKey}):`, err);
            pdfDoc.setFontSize(12);
            pdfDoc.text(`Unable to render optimization result for ${methodName}.`, margin, 40);
            
            // Try to capture a screenshot of just the essential info if full card fails
            try {
              const essentialInfo = card.querySelector('[class*="MuiCardContent-root"]');
              if (essentialInfo) {
                const essentialCanvas = await html2canvas(essentialInfo as HTMLElement, {
                  ...commonCanvasOptions,
                  scale: 1.2
                });
                const essentialImgData = essentialCanvas.toDataURL('image/png');
                pdfDoc.addImage(
                  essentialImgData, 
                  'PNG', 
                  margin, 
                  50, 
                  pageWidth - (margin * 2), 
                  (essentialCanvas.height * (pageWidth - (margin * 2))) / essentialCanvas.width
                );
                console.log(`Rendered essential info for ${methodName} as fallback`);
              }
            } catch (fallbackErr) {
              console.error(`Even fallback rendering failed for ${methodName}:`, fallbackErr);
            }
          }
        }
      } else {
        console.error('Results container not found for PDF generation');
      }

      // ----------- 3. CUMULATIVE RETURNS CHART -----------
      if (!isTechnicalReport) {
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
      }

      // ----------- 4. YEARLY RETURNS TABLE -----------
      if (!isTechnicalReport) {
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
      }

      // ----------- 5. COVARIANCE HEATMAP -----------
      if (!isTechnicalReport && optimizationResult.covariance_heatmap) {
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

  // Add a helper function to calculate final cumulative return for technical optimization
  const calculateFinalCumulativeReturn = (returns: (number | null)[] | undefined): number | null => {
    if (!returns || returns.length === 0) return null;
    // Return the last non-null value in the array
    for (let i = returns.length - 1; i >= 0; i--) {
      if (returns[i] !== null) return returns[i];
    }
    return null;
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
      
      {/* Technical Indicators Section - Only shown when TECHNICAL is selected */}
      {isTechnicalOptimizationSelected && (
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
            Select Technical Indicators
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
          
          <Typography variant="body2" style={{ marginBottom: '16px' }}>
            Technical indicator optimization uses cross-sectional signal z-scores to form portfolios. 
            Select at least 2 indicators to proceed.
          </Typography>
          
          <Grid container spacing={2} style={{ marginBottom: '16px' }}>
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Indicator Type</InputLabel>
                <Select
                  value={selectedIndicatorName}
                  onChange={(e) => setSelectedIndicatorName(e.target.value as TechnicalIndicatorType)}
                  label="Indicator Type"
                >
                  {Object.values(TechnicalIndicatorType).map((type) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={3}>
              <FormControl fullWidth variant="outlined">
                <InputLabel>Window Size</InputLabel>
                <Select
                  value={selectedIndicatorWindow}
                  onChange={(e) => setSelectedIndicatorWindow(e.target.value)}
                  label="Window Size"
                >
                  {selectedIndicatorName && indicatorWindowOptions[selectedIndicatorName]?.map((window: number) => (
                    <MenuItem key={window} value={window.toString()}>
                      {window}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            {selectedIndicatorName && (selectedIndicatorName === TechnicalIndicatorType.SUPERTREND) && (
              <Grid item xs={12} sm={3}>
                <TextField
                  fullWidth
                  label="Multiplier"
                  type="number"
                  value={selectedIndicatorMult}
                  onChange={(e) => setSelectedIndicatorMult(e.target.value)}
                  variant="outlined"
                  InputProps={{ inputProps: { min: 0.1, step: 0.1 } }}
                />
              </Grid>
            )}
            <Grid item xs={12} sm={2}>
              <Button
                fullWidth
                variant="contained"
                color="primary"
                onClick={handleAddIndicator}
                style={{ height: '56px' }}
              >
                Add
              </Button>
            </Grid>
          </Grid>
          
          {/* Display selected indicators */}
          <div>
            <Typography variant="subtitle1" style={{ marginBottom: '8px', fontWeight: 500 }}>
              Selected Indicators ({selectedIndicators.length})
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {selectedIndicators.map((indicator, idx) => (
                <Chip
                  key={idx}
                  label={`${indicator.name} (Window: ${indicator.window}${indicator.mult ? `, Mult: ${indicator.mult}` : ''})`}
                  onDelete={() => handleRemoveIndicator(idx)}
                  className="m-1"
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Stack>
            {needsMinimum2Indicators && (
              <Typography color="error" style={{ marginTop: '8px', fontSize: '0.875rem' }}>
                Please select at least 2 technical indicators to proceed with Technical Optimization.
              </Typography>
            )}
          </div>
        </div>
      )}

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
            
            {/* Date Range and Risk-Free Rate Information */}
            <Box sx={{ mb: 3 }}>
              {/* For non-technical optimization or mixed optimization */}
              {!isTechnicalOptimizationResult && (
                <Card sx={{ mb: 2, backgroundColor: '#f5f9ff', border: '1px solid #d0e8fa' }}>
                  <CardContent>
                    <Typography variant="h6" sx={{ mb: 1, fontWeight: 600, fontSize: '1.1rem' }}>
                      Return-Based Optimization Parameters
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body1">
                          <strong>Period:</strong> {formatDate(optimizationResult.start_date)} to {formatDate(optimizationResult.end_date)}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body1">
                          <strong>Risk-Free Rate:</strong> {((optimizationResult.risk_free_rate || 0) * 100).toFixed(2)}%
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              )}

              {/* For technical-only optimization or mixed optimization with technical */}
              {(isTechnicalOptimizationResult || (optimizationResult.technical_start_date && optimizationResult.technical_end_date)) && (
                <Card sx={{ backgroundColor: '#f0f8ff', border: '1px solid #b3d8ff' }}>
                  <CardContent>
                    <Typography variant="h6" sx={{ mb: 1, fontWeight: 600, fontSize: '1.1rem' }}>
                      Technical Optimization Parameters
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body1">
                          <strong>Period:</strong> {formatDate(optimizationResult.technical_start_date || optimizationResult.start_date)} to {formatDate(optimizationResult.technical_end_date || optimizationResult.end_date)}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body1">
                          <strong>Risk-Free Rate:</strong> {((optimizationResult.technical_risk_free_rate || optimizationResult.risk_free_rate || 0) * 100).toFixed(2)}%
                        </Typography>
                      </Grid>
                      <Grid item xs={12}>
                        <Typography variant="body2">
                          Technical indicator optimization uses cross-sectional signal z-scores to form portfolios.
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              )}
            </Box>
            
            {/* Technical Optimization Badge */}
            {isTechnicalOptimizationResult && (
              <Card style={{ marginBottom: '20px', backgroundColor: '#f0fff4', border: '1px solid #8eedc7' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <CheckCircleOutline style={{ marginRight: '10px', color: '#38a169' }} />
                    <div>
                      <Typography variant="subtitle1" fontWeight="bold" color="#38a169">
                        Technical Indicator Optimization
                      </Typography>
                      <Typography variant="body2">
                        This portfolio is optimized using technical indicators rather than historical returns.
                        It uses cross-sectional signal z-scores to form portfolios based on the selected indicators.
                      </Typography>
                    </div>
                  </Box>
                </CardContent>
              </Card>
            )}
            
            {/* Method Cards */}
            {Object.entries(optimizationResult.results).map(([methodKey, methodData], index) => {
              if (!methodData) return null;

              // Check if this is a technical optimization result (contains ONLY technical method)
              const isTechnicalMethod = methodKey === "TECHNICAL";
              
              return (
                <Card key={methodKey} sx={{ mb: 3, boxShadow: 3 }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom sx={{ fontSize: '1.3rem', fontWeight: 600 }}>
                      {getMethodDisplayName(methodKey)} Portfolio
                    </Typography>
                    
                    <Grid container spacing={3}>
                      {/* Left Column: Metrics, Advanced Metrics, and Weights */}
                      <Grid item xs={12} md={4}>
                        {/* Performance Metrics */}
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
                                  <TableCell>Sharpe Ratio</TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.sharpe.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>Sortino Ratio</TableCell>
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
                                
                                {/* Add cumulative return for technical optimization */}
                                {methodKey === "TECHNICAL" && optimizationResult?.cumulative_returns[methodKey] && (
                                  <TableRow>
                                    <TableCell>Cumulative Return</TableCell>
                                    <TableCell align="right" style={getReturnCellStyle(calculateFinalCumulativeReturn(optimizationResult.cumulative_returns[methodKey]) || 0)}>
                                      {calculateFinalCumulativeReturn(optimizationResult.cumulative_returns[methodKey]) 
                                        ? `${((calculateFinalCumulativeReturn(optimizationResult.cumulative_returns[methodKey]) as number - 1) * 100).toFixed(2)}%` 
                                        : 'N/A'}
                                    </TableCell>
                                  </TableRow>
                                )}
                                
                                {/* Only hide specific beta-related metrics for TECHNICAL method */}
                                {methodKey !== "TECHNICAL" ? (
                                  <>
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
                                  </>
                                ) : (
                                  // Add a special message for technical method explaining why beta metrics are omitted
                                  <TableRow>
                                    <TableCell colSpan={2} sx={{ color: 'text.secondary', fontSize: '0.85rem', fontStyle: 'italic' }}>
                                      Note: Beta-related metrics (Beta, Alpha, R-Squared) are not applicable for technical optimization as it does not use CAPM model.
                                    </TableCell>
                                  </TableRow>
                                )}
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
                                  <TableCell>Skewness</TableCell>
                                  <TableCell align="right" style={methodData.performance.skewness > 0 ? { color: 'green' } : { color: 'red' }}>
                                    {methodData.performance.skewness.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>Kurtosis</TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.kurtosis.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                
                                {/* Show Treynor for all methods including technical optimization */}
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
                                
                                {/* Add more general risk metrics for all methods */}
                                <TableRow>
                                  <TableCell>
                                    Calmar Ratio
                                    <Tooltip title="The Calmar Ratio is the annualized return divided by the maximum drawdown. It measures return per unit of downside risk.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.calmar_ratio.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Omega Ratio
                                    <Tooltip title="The Omega Ratio measures the probability-weighted ratio of gains versus losses for a given return threshold. Higher values are better.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {Number.isFinite(methodData.performance.omega_ratio) ? methodData.performance.omega_ratio.toFixed(3) : '∞'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Ulcer Index
                                    <Tooltip title="The Ulcer Index measures the depth and duration of drawdowns in asset prices. Lower values indicate less drawdown risk.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.ulcer_index.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                              </>
                            )}
                          </TableBody>
                        </Table>
                        
                        {/* Advanced Beta and Cross-Moment Metrics with NEW tag */}
                        {methodKey !== "TECHNICAL" && (
                          <Box sx={{ mt: 3 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <Typography variant="h6" sx={{ fontSize: '1.1rem', fontWeight: 600, mr: 1 }}>
                                Advanced Beta & Cross-Moment Metrics
                              </Typography>
                              <Chip 
                                label="NEW" 
                                size="small" 
                                sx={{ 
                                  backgroundColor: '#4caf50', 
                                  color: 'white', 
                                  fontWeight: 'bold',
                                  fontSize: '0.7rem'
                                }} 
                              />
                            </Box>
                            <Table size="small">
                              <TableBody>
                                <TableRow>
                                  <TableCell>
                                    Welch Beta
                                    <Tooltip title="Alternative beta calculation that filters out unusual returns using a trimming approach specific to how extreme the market returns were. Robust to outliers and extreme market conditions.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.welch_beta !== undefined && !isNaN(methodData.performance.welch_beta) ? methodData.performance.welch_beta.toFixed(3) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Semi Beta
                                    <Tooltip title="Downside beta that only considers periods when the market return is below a threshold (usually zero). Measures sensitivity to market downturns specifically.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.semi_beta !== undefined && !isNaN(methodData.performance.semi_beta) ? methodData.performance.semi_beta.toFixed(3) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Blume-Adjusted Beta
                                    <Tooltip title="Beta adjusted using Blume's technique which addresses the tendency of beta to revert to 1.0 over time. Calculated as: 0.67 × Beta + 0.33 × 1.0">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.blume_adjusted_beta.toFixed(3)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Beta p-value
                                    <Tooltip title="Statistical significance of the beta estimate. Lower values indicate higher confidence in the beta value (reject null hypothesis of β=0).">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.beta_pvalue.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Coskewness
                                    <Tooltip title="Measures the relationship between portfolio returns and squared market returns. Negative values suggest the portfolio tends to have negative returns when market volatility increases.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.coskewness !== undefined && !isNaN(methodData.performance.coskewness) ? methodData.performance.coskewness.toFixed(4) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Cokurtosis
                                    <Tooltip title="Fourth cross-moment measuring the relationship between portfolio returns and extreme market returns. A high positive value indicates portfolio returns amplify extreme market movements.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.cokurtosis !== undefined && !isNaN(methodData.performance.cokurtosis) ? methodData.performance.cokurtosis.toFixed(4) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Entropy
                                    <Tooltip title="Shannon entropy of the return distribution. Higher values indicate more dispersed returns.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.entropy.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Gini Mean Difference
                                    <Tooltip title="A measure of variability based on the average absolute difference between all pairs of returns. Higher values indicate greater dispersion.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {methodData.performance.gini_mean_difference.toFixed(4)}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    DaR (95%)
                                    <Tooltip title="Drawdown at Risk - the 95th percentile of the drawdown distribution. Represents the worst 5% of drawdowns.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(-methodData.performance.dar_95)}>
                                    {(methodData.performance.dar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    CDaR (95%)
                                    <Tooltip title="Conditional Drawdown at Risk - the expected drawdown given that drawdown exceeds DaR. Average of the worst 5% of drawdowns.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(-methodData.performance.cdar_95)}>
                                    {(methodData.performance.cdar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    EVaR (95%)
                                    <Tooltip title="Entropic Value at Risk - a coherent risk measure that uses an exponential transformation. More sensitive to tail risk than traditional VaR.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right" style={getReturnCellStyle(methodData.performance.evar_95)}>
                                    {(methodData.performance.evar_95 * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Upside Potential Ratio
                                    <Tooltip title="Ratio of upside potential to downside risk. Measures the probability-weighted upside returns relative to downside volatility.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {Number.isFinite(methodData.performance.upside_potential_ratio) ? methodData.performance.upside_potential_ratio.toFixed(4) : '∞'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    M² (Modigliani Risk-Adjusted Performance)
                                    <Tooltip title="Risk-adjusted return measure that adjusts portfolio returns to the same risk level as the benchmark. Expressed as percentage return.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {(methodData.performance.modigliani_risk_adjusted_performance * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    Sterling Ratio
                                    <Tooltip title="Risk-adjusted return measure calculated as CAGR divided by (Average Annual Drawdown - 10%). Higher values are better.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {Number.isFinite(methodData.performance.sterling_ratio) && !isNaN(methodData.performance.sterling_ratio) ? methodData.performance.sterling_ratio.toFixed(3) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                                <TableRow>
                                  <TableCell>
                                    V2 Ratio
                                    <Tooltip title="Relative risk-adjusted return measure that compares portfolio performance to benchmark on a drawdown-adjusted basis.">
                                      <InfoOutlined fontSize="small" style={{ marginLeft: '4px', verticalAlign: 'middle', cursor: 'help' }} />
                                    </Tooltip>
                                  </TableCell>
                                  <TableCell align="right">
                                    {Number.isFinite(methodData.performance.v2_ratio) && !isNaN(methodData.performance.v2_ratio) ? methodData.performance.v2_ratio.toFixed(4) : 'N/A'}
                                  </TableCell>
                                </TableRow>
                              </TableBody>
                            </Table>
                          </Box>
                        )}
                        
                        {/* Portfolio Weights Table */}
                        <Box sx={{ mt: 3 }}>
                          <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                            Portfolio Weights
                          </Typography>
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell sx={{ fontWeight: 'bold' }}>Stock</TableCell>
                                <TableCell align="right" sx={{ fontWeight: 'bold' }}>Weight</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(methodData.weights)
                                .sort(([,a], [,b]) => b - a) // Sort by weight descending
                                .map(([ticker, weight]) => (
                                  <TableRow key={ticker}>
                                    <TableCell>{ticker}</TableCell>
                                    <TableCell align="right" sx={{ fontWeight: weight > 0.1 ? 'bold' : 'normal' }}>
                                      {(weight * 100).toFixed(2)}%
                                    </TableCell>
                                  </TableRow>
                                ))}
                            </TableBody>
                          </Table>
                        </Box>
                      </Grid>
                      
                      {/* Right Column: Visualizations */}
                      <Grid item xs={12} md={8}>
                        {/* Returns Distribution */}
                        {methodData.returns_dist && (
                          <Box>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              {methodKey === "TECHNICAL" ? "Technical Strategy Returns Distribution" : "Returns Distribution"}
                            </Typography>
                            <Box sx={{ border: '1px solid #eaeaea', borderRadius: '4px', overflow: 'hidden' }}>
                              <ImageComponent base64String={methodData.returns_dist} altText="Returns Distribution" />
                            </Box>
                            {methodKey === "TECHNICAL" && (
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                                This distribution shows the daily returns of the technical indicator-based portfolio.
                              </Typography>
                            )}
                          </Box>
                        )}
                        
                        {/* Maximum Drawdown */}
                        {methodData.max_drawdown_plot && (
                          <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', fontWeight: 600 }}>
                              {methodKey === "TECHNICAL" ? "Technical Strategy Drawdown" : "Maximum Drawdown"}
                            </Typography>
                            <Box sx={{ border: '1px solid #eaeaea', borderRadius: '4px', overflow: 'hidden' }}>
                              <ImageComponent base64String={methodData.max_drawdown_plot} altText="Maximum Drawdown" />
                            </Box>
                            {methodKey === "TECHNICAL" && (
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                                This chart compares the drawdown of the technical indicator-based portfolio against the benchmark.
                              </Typography>
                            )}
                          </Box>
                        )}
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              );
            })}
            
            {/* Cumulative Returns Chart */}
            {optimizationResult && !isTechnicalOptimizationResult && optimizationResult.dates && optimizationResult.cumulative_returns && (
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
            {optimizationResult && optimizationResult.results && 
              // Filter to check if there are any non-TECHNICAL methods with rolling betas
              Object.entries(optimizationResult.results)
                .filter(([methodKey, methodData]) => methodKey !== "TECHNICAL" && methodData && methodData.rolling_betas)
                .length > 0 && (
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
                      data={prepareBetaChartData(optimizationResult)}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: { position: 'top' as const },
                          tooltip: { mode: 'index' as const, intersect: false }
                        },
                        scales: {
                          y: {
                            display: true,
                            title: { display: true, text: 'Beta Value' },
                          }
                        }
                      }}
                    />
                  </div>
                </CardContent>
              </Card>
            )}
            
            {/* Yearly Stock Returns */}
            {optimizationResult && optimizationResult.stock_yearly_returns && Object.keys(optimizationResult.stock_yearly_returns).length > 0 && (
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
            
            {/* Covariance Heatmap - Only show for non-TECHNICAL methods */}
            {optimizationResult && !selectedAlgorithms.every(algo => algo.value === "TECHNICAL") && optimizationResult.covariance_heatmap && (
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