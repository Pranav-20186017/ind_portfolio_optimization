import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { StockData, StockOption, PortfolioOptimizationResponse } from '../types';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import axios from 'axios';
import Fuse from 'fuse.js';
import debounce from 'lodash/debounce';

// Import Chart.js components
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

// Component to display base64 image
const ImageComponent = ({ base64String, altText }: { base64String: string, altText: string }) => (
    <div className="image-container">
        {base64String ? (
            <img
                src={`data:image/png;base64,${base64String}`}
                alt={altText}
                style={{ width: '100%', height: 'auto' }}
            />
        ) : (
            <p>No image available</p>
        )}
    </div>
);

const HomePage: React.FC = () => {
    const [stockData, setStockData] = useState<StockData>({});
    const [options, setOptions] = useState<StockOption[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [selectedStocks, setSelectedStocks] = useState<StockOption[]>([]);
    const [optimizationResult, setOptimizationResult] = useState<PortfolioOptimizationResponse | null>(null);
    const [filteredOptions, setFilteredOptions] = useState<StockOption[]>([]);

    useEffect(() => {
        // Fetch stock data from public/stock_data.json
        const fetchStockData = async () => {
            try {
                const res = await fetch('/stock_data.json');
                if (!res.ok) {
                    throw new Error(`HTTP error! Status: ${res.status}`);
                }
                const data: StockData = await res.json();
                setStockData(data);

                // Transform stockData into an array of options
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

    // Initialize Fuse.js for fuzzy search
    const fuse = useMemo(() => {
        return new Fuse(options, {
            keys: ['ticker', 'name'],
            threshold: 0.3,
        });
    }, [options]);

    // Debounced filtering function
    const debouncedFilter = useCallback(
        debounce((input: string) => {
            if (input === '') {
                setFilteredOptions([]);
                return;
            }
            const results = fuse.search(input);
            setFilteredOptions(results.slice(0, 50).map((result) => result.item));
        }, 300),
        [fuse]
    );

    useEffect(() => {
        debouncedFilter(inputValue);
    }, [inputValue, debouncedFilter]);

    const handleAddStock = (event: any, newValue: StockOption | null) => {
        if (newValue) {
            const isDuplicate = selectedStocks.some(
                (stock) =>
                    stock.ticker === newValue.ticker && stock.exchange === newValue.exchange
            );
            if (!isDuplicate) {
                setSelectedStocks([...selectedStocks, newValue]);
            }
        }
        setInputValue('');
    };

    const handleRemoveStock = (stockToRemove: StockOption) => {
        setSelectedStocks(
            selectedStocks.filter(
                (stock) =>
                    !(
                        stock.ticker === stockToRemove.ticker &&
                        stock.exchange === stockToRemove.exchange
                    )
            )
        );
    };

    const handleSubmit = async () => {
        const dataToSend = {
            stocks: selectedStocks.map((stock) => ({
                ticker: stock.ticker,
                exchange: stock.exchange,
            })),
        };

        try {
            const response = await axios.post('https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/optimize', dataToSend);
            const result = response.data as PortfolioOptimizationResponse;
            setOptimizationResult(result);
        } catch (error) {
            console.error('API Error:', error);
        }
    };

    const handleReset = () => {
        setSelectedStocks([]);
        setInputValue('');
        setOptimizationResult(null);
    };

    const formatDate = (dateString: string) => {
        const options: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'short', day: 'numeric' };
        return new Date(dateString).toLocaleDateString(undefined, options);
    };

    const prepareChartData = (result: PortfolioOptimizationResponse) => {
        const labels = result.dates.map((dateStr) => new Date(dateStr).toLocaleDateString());

        const datasets = [];

        if (result.cumulative_returns.MVO && result.cumulative_returns.MVO.length > 0) {
            datasets.push({
                label: 'MVO Portfolio',
                data: result.cumulative_returns.MVO,
                borderColor: 'blue',
                fill: false,
            });
        }

        if (result.cumulative_returns.MinVol && result.cumulative_returns.MinVol.length > 0) {
            datasets.push({
                label: 'MinVol Portfolio',
                data: result.cumulative_returns.MinVol,
                borderColor: 'green',
                fill: false,
            });
        }
        if (result.cumulative_returns.MaxQuadraticUtility && result.cumulative_returns.MaxQuadraticUtility.length > 0) {
            datasets.push({
                label: 'Max Quadratic Utility Portfolio',
                data: result.cumulative_returns.MaxQuadraticUtility,
                borderColor: 'purple',  // Use a distinct color for this portfolio
                fill: false,
            });
        }


        if (result.nifty_returns && result.nifty_returns.length > 0) {
            datasets.push({
                label: 'Nifty Index',
                data: result.nifty_returns,
                borderColor: 'red',
                fill: false,
            });
        }

        return {
            labels,
            datasets,
        };
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
        <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Stock Search</h1>
            <Autocomplete
                options={filteredOptions}
                getOptionLabel={(option) => `${option.ticker} - ${option.name} (${option.exchange})`}
                onChange={handleAddStock}
                inputValue={inputValue}
                onInputChange={(event, newInputValue) => setInputValue(newInputValue)}
                renderInput={(params) => (
                    <TextField {...params} label="Search Stock" variant="outlined" />
                )}
                style={{ width: '100%', maxWidth: 600 }}
                openOnFocus={false}
                filterOptions={(options) => options}
                clearOnBlur={false}
                value={null}
            />
            <div className="mt-6">
                <h2 className="text-xl font-semibold mb-2">Selected Stocks</h2>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                    {selectedStocks.map((stock, index) => (
                        <Chip
                            key={index}
                            label={`${stock.ticker} (${stock.exchange})`}
                            onDelete={() => handleRemoveStock(stock)}
                            className="m-1"
                        />
                    ))}
                </Stack>
            </div>
            <div className="mt-6">
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSubmit}
                    className="mr-4"
                    disabled={selectedStocks.length === 0}
                >
                    Submit
                </Button>
                <Button variant="outlined" color="secondary" onClick={handleReset}>
                    Reset
                </Button>
            </div>

            {/* Optimization Results */}
            <div className="mt-6">
                {optimizationResult && (
                    <div className="results-container">
                        {/* Results Section */}
                        <div className="results">
                            <h2 className="text-xl font-semibold mb-2">Optimization Results</h2>
                            <p>Data Time Period: {formatDate(optimizationResult.start_date)} to {formatDate(optimizationResult.end_date)}</p>

                            {optimizationResult.MVO && (
                                <div className="result-with-plot">
                                    <div className="result-details">
                                        <h3 className="text-lg font-semibold">Mean-Variance Optimization (MVO)</h3>
                                        <p>Expected Return: {optimizationResult.MVO.performance.expected_return.toFixed(4)}</p>
                                        <p>Volatility: {optimizationResult.MVO.performance.volatility.toFixed(4)}</p>
                                        <p>Sharpe Ratio: {optimizationResult.MVO.performance.sharpe.toFixed(4)}</p>
                                        <h4 className="font-semibold">Weights:</h4>
                                        <ul>
                                            {Object.entries(optimizationResult.MVO.weights).map(([ticker, weight]) => (
                                                <li key={ticker}>{ticker}: {(weight * 100).toFixed(2)}%</li>
                                            ))}
                                        </ul>
                                    </div>
                                    {/* MVO Plot */}
                                    <ImageComponent base64String={optimizationResult.MVO.returns_dist} altText="MVO Portfolio Distribution" />
                                </div>
                            )}

                            {optimizationResult.MinVol && (
                                <div className="result-with-plot">
                                    <div className="result-details">
                                        <h3 className="text-lg font-semibold">Minimum Volatility Portfolio</h3>
                                        <p>Expected Return: {optimizationResult.MinVol.performance.expected_return.toFixed(4)}</p>
                                        <p>Volatility: {optimizationResult.MinVol.performance.volatility.toFixed(4)}</p>
                                        <p>Sharpe Ratio: {optimizationResult.MinVol.performance.sharpe.toFixed(4)}</p>
                                        <h4 className="font-semibold">Weights:</h4>
                                        <ul>
                                            {Object.entries(optimizationResult.MinVol.weights).map(([ticker, weight]) => (
                                                <li key={ticker}>{ticker}: {(weight * 100).toFixed(2)}%</li>
                                            ))}
                                        </ul>
                                    </div>
                                    {/* Min Vol Plot */}
                                    <ImageComponent base64String={optimizationResult.MinVol.returns_dist} altText="MinVol Portfolio Distribution" />
                                </div>
                            )}

                            {optimizationResult.MaxQuadraticUtility && (
                                <div className="result-with-plot">
                                    <div className="result-details">
                                        <h3 className="text-lg font-semibold">Max Quadratic Utility Portfolio</h3>
                                        <p>Expected Return: {optimizationResult.MaxQuadraticUtility.performance.expected_return.toFixed(4)}</p>
                                        <p>Volatility: {optimizationResult.MaxQuadraticUtility.performance.volatility.toFixed(4)}</p>
                                        <p>Sharpe Ratio: {optimizationResult.MaxQuadraticUtility.performance.sharpe.toFixed(4)}</p>
                                        <h4 className="font-semibold">Weights:</h4>
                                        <ul>
                                            {Object.entries(optimizationResult.MaxQuadraticUtility.weights).map(([ticker, weight]) => (
                                                <li key={ticker}>{ticker}: {(weight * 100).toFixed(2)}%</li>
                                            ))}
                                        </ul>
                                    </div>
                                    {/* Max Quadratic Utility Plot */}
                                    <ImageComponent base64String={optimizationResult.MaxQuadraticUtility.returns_dist} altText="Max Quadratic Utility Portfolio Distribution" />
                                </div>
                            )}


                            {/* Cumulative Returns Chart */}
                            <div className="mt-6">
                                <h2 className="text-xl font-semibold mb-2">Cumulative Returns Over Time</h2>
                                <Line data={prepareChartData(optimizationResult)} options={chartOptions} />
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Styles */}
            <style jsx>{`
                .results-container {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }

                .result-with-plot {
                    display: flex;
                    gap: 20px;
                    align-items: flex-start;
                }

                .result-details {
                    flex: 1;
                }

                .image-container {
                    flex: 1;
                    border: 1px solid #ccc;
                    padding: 10px;
                    background-color: #f9f9f9;
                }
            `}</style>
        </div>
    );
};

export default HomePage;
