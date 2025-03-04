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
import { BorderColor } from '@mui/icons-material';

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Legend, Tooltip);

// Component to display base64-encoded images
const ImageComponent = ({ base64String, altText }: { base64String: string; altText: string }) => (
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

    // ----------------------------------------
    // 1) Fetch stock data
    // ----------------------------------------
    useEffect(() => {
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

    // ----------------------------------------
    // 2) Fuzzy search with Fuse.js
    // ----------------------------------------
    const fuse = useMemo(() => {
        return new Fuse(options, {
            keys: ['ticker', 'name'],
            threshold: 0.3,
        });
    }, [options]);

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

    // ----------------------------------------
    // 3) Adding/removing stocks
    // ----------------------------------------
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

    // ----------------------------------------
    // 4) Submit to the backend
    // ----------------------------------------
    const handleSubmit = async () => {
        const dataToSend = {
            stocks: selectedStocks.map((stock) => ({
                ticker: stock.ticker,
                exchange: stock.exchange,
            })),
        };

        try {
            const response = await axios.post(
                'https://vgb7u5iqyb.execute-api.us-east-2.amazonaws.com/optimize',
                dataToSend
            );
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

    // ----------------------------------------
    // 5) Date formatting
    // ----------------------------------------
    const formatDate = (dateString: string) => {
        const options: Intl.DateTimeFormatOptions = { year: 'numeric', month: 'short', day: 'numeric' };
        return new Date(dateString).toLocaleDateString(undefined, options);
    };

    // ----------------------------------------
    // 6) Prepare Chart.js data
    // ----------------------------------------
    const prepareChartData = (result: PortfolioOptimizationResponse) => {
        const labels = result.dates.map((dateStr) => new Date(dateStr).toLocaleDateString());
        const datasets = [];

        if (result.cumulative_returns.MVO?.length) {
            datasets.push({
                label: 'MVO Portfolio',
                data: result.cumulative_returns.MVO,
                borderColor: 'blue',
                fill: false,
            });
        }
        if (result.cumulative_returns.MinVol?.length) {
            datasets.push({
                label: 'MinVol Portfolio',
                data: result.cumulative_returns.MinVol,
                borderColor: 'green',
                fill: false,
            });
        }
        if (result.cumulative_returns.MaxQuadraticUtility?.length) {
            datasets.push({
                label: 'Max Quadratic Utility',
                data: result.cumulative_returns.MaxQuadraticUtility,
                borderColor: 'purple',
                fill: false,
            });
        }
        if (result.cumulative_returns.EquiWeighted?.length) {
            datasets.push({
                label: 'Equi Weighted',
                data: result.cumulative_returns.EquiWeighted,
                borderColor: 'orange',
                fill:false,
            });
        }
        if (result.cumulative_returns.CriticalLineAlgorithm?.length) {
            datasets.push({
                label: 'Critical Line Algorithm',
                data: result.cumulative_returns.CriticalLineAlgorithm,
                borderColor: 'magenta',
                fill:false,
            });
        }
        if (result.nifty_returns?.length) {
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

    // ----------------------------------------
    // 7) Render
    // ----------------------------------------
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
                filterOptions={(x) => x}
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
                        <h2 className="text-xl font-semibold mb-2">Optimization Results</h2>
                        <p>
                            Data Time Period:{' '}
                            {formatDate(optimizationResult.start_date)} to{' '}
                            {formatDate(optimizationResult.end_date)}
                        </p>

                        {/* MVO */}
                        {optimizationResult.MVO && (
                            <div className="result-with-plot">
                                <div className="result-details">
                                    <h3 className="text-lg font-semibold">MVO (Mean-Variance Optimization)</h3>
                                    <p>Expected Return: {optimizationResult.MVO.performance.expected_return.toFixed(4)}</p>
                                    <p>Volatility: {optimizationResult.MVO.performance.volatility.toFixed(4)}</p>
                                    <p>Sharpe Ratio: {optimizationResult.MVO.performance.sharpe.toFixed(4)}</p>

                                    <p>Sortino Ratio: {optimizationResult.MVO.performance.sortino.toFixed(4)}</p>
                                    <p>Max Drawdown: {(optimizationResult.MVO.performance.max_drawdown * 100).toFixed(2)}%</p>
                                    <p>RoMaD: {optimizationResult.MVO.performance.romad.toFixed(4)}</p>

                                    <p>VaR 95%: {(optimizationResult.MVO.performance.var_95 * 100).toFixed(2)}%</p>
                                    <p>CVaR 95%: {(optimizationResult.MVO.performance.cvar_95 * 100).toFixed(2)}%</p>
                                    <p>VaR 90%: {(optimizationResult.MVO.performance.var_90 * 100).toFixed(2)}%</p>
                                    <p>CVaR 90%: {(optimizationResult.MVO.performance.cvar_90 * 100).toFixed(2)}%</p>
                                    <p>CAGR: {(optimizationResult.MVO.performance.cagr * 100).toFixed(2)}%</p>

                                    <h4 className="font-semibold mt-2">Weights:</h4>
                                    <ul>
                                        {Object.entries(optimizationResult.MVO.weights).map(([ticker, weight]) => (
                                            <li key={ticker}>
                                                {ticker}: {(weight * 100).toFixed(2)}%
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="plots-container">
                                    {/* Distribution Plot */}
                                    <ImageComponent
                                        base64String={optimizationResult.MVO.returns_dist}
                                        altText="MVO Portfolio Distribution"
                                    />
                                    {/* Drawdown Plot */}
                                    <ImageComponent
                                        base64String={optimizationResult.MVO.max_drawdown_plot}
                                        altText="MVO Portfolio Drawdown"
                                    />
                                </div>
                            </div>
                        )}

                        {/* MinVol */}
                        {optimizationResult.MinVol && (
                            <div className="result-with-plot">
                                <div className="result-details">
                                    <h3 className="text-lg font-semibold">MinVol (Minimum Volatility)</h3>
                                    <p>Expected Return: {optimizationResult.MinVol.performance.expected_return.toFixed(4)}</p>
                                    <p>Volatility: {optimizationResult.MinVol.performance.volatility.toFixed(4)}</p>
                                    <p>Sharpe Ratio: {optimizationResult.MinVol.performance.sharpe.toFixed(4)}</p>

                                    <p>Sortino Ratio: {optimizationResult.MinVol.performance.sortino.toFixed(4)}</p>
                                    <p>Max Drawdown: {(optimizationResult.MinVol.performance.max_drawdown * 100).toFixed(2)}%</p>
                                    <p>RoMaD: {optimizationResult.MinVol.performance.romad.toFixed(4)}</p>

                                    <p>VaR 95%: {(optimizationResult.MinVol.performance.var_95 * 100).toFixed(2)}%</p>
                                    <p>CVaR 95%: {(optimizationResult.MinVol.performance.cvar_95 * 100).toFixed(2)}%</p>
                                    <p>VaR 90%: {(optimizationResult.MinVol.performance.var_90 * 100).toFixed(2)}%</p>
                                    <p>CVaR 90%: {(optimizationResult.MinVol.performance.cvar_90 * 100).toFixed(2)}%</p>
                                    <p>CAGR: {(optimizationResult.MinVol.performance.cagr * 100).toFixed(2)}%</p>

                                    <h4 className="font-semibold mt-2">Weights:</h4>
                                    <ul>
                                        {Object.entries(optimizationResult.MinVol.weights).map(([ticker, weight]) => (
                                            <li key={ticker}>
                                                {ticker}: {(weight * 100).toFixed(2)}%
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="plots-container">
                                    <ImageComponent
                                        base64String={optimizationResult.MinVol.returns_dist}
                                        altText="MinVol Portfolio Distribution"
                                    />
                                    <ImageComponent
                                        base64String={optimizationResult.MinVol.max_drawdown_plot}
                                        altText="MinVol Portfolio Drawdown"
                                    />
                                </div>
                            </div>
                        )}

                        {/* MaxQuadraticUtility */}
                        {optimizationResult.MaxQuadraticUtility && (
                            <div className="result-with-plot">
                                <div className="result-details">
                                    <h3 className="text-lg font-semibold">Max Quadratic Utility</h3>
                                    <p>Expected Return: {optimizationResult.MaxQuadraticUtility.performance.expected_return.toFixed(4)}</p>
                                    <p>Volatility: {optimizationResult.MaxQuadraticUtility.performance.volatility.toFixed(4)}</p>
                                    <p>Sharpe Ratio: {optimizationResult.MaxQuadraticUtility.performance.sharpe.toFixed(4)}</p>

                                    <p>Sortino Ratio: {optimizationResult.MaxQuadraticUtility.performance.sortino.toFixed(4)}</p>
                                    <p>Max Drawdown: {(optimizationResult.MaxQuadraticUtility.performance.max_drawdown * 100).toFixed(2)}%</p>
                                    <p>RoMaD: {optimizationResult.MaxQuadraticUtility.performance.romad.toFixed(4)}</p>

                                    <p>VaR 95%: {(optimizationResult.MaxQuadraticUtility.performance.var_95 * 100).toFixed(2)}%</p>
                                    <p>CVaR 95%: {(optimizationResult.MaxQuadraticUtility.performance.cvar_95 * 100).toFixed(2)}%</p>
                                    <p>VaR 90%: {(optimizationResult.MaxQuadraticUtility.performance.var_90 * 100).toFixed(2)}%</p>
                                    <p>CVaR 90%: {(optimizationResult.MaxQuadraticUtility.performance.cvar_90 * 100).toFixed(2)}%</p>
                                    <p>CAGR: {(optimizationResult.MaxQuadraticUtility.performance.cagr * 100).toFixed(2)}%</p>

                                    <h4 className="font-semibold mt-2">Weights:</h4>
                                    <ul>
                                        {Object.entries(optimizationResult.MaxQuadraticUtility.weights).map(([ticker, weight]) => (
                                            <li key={ticker}>
                                                {ticker}: {(weight * 100).toFixed(2)}%
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="plots-container">
                                    <ImageComponent
                                        base64String={optimizationResult.MaxQuadraticUtility.returns_dist}
                                        altText="Max Quadratic Utility Distribution"
                                    />
                                    <ImageComponent
                                        base64String={optimizationResult.MaxQuadraticUtility.max_drawdown_plot}
                                        altText="Max Quadratic Utility Drawdown"
                                    />
                                </div>
                            </div>
                        )}

                        {/* EquiWeighted */}
                        {optimizationResult.EquiWeighted && (
                            <div className="result-with-plot">
                                <div className="result-details">
                                    <h3 className="text-lg font-semibold">EquiWeighted</h3>
                                    <p>Expected Return: {optimizationResult.EquiWeighted.performance.expected_return.toFixed(4)}</p>
                                    <p>Volatility: {optimizationResult.EquiWeighted.performance.volatility.toFixed(4)}</p>
                                    <p>Sharpe Ratio: {optimizationResult.EquiWeighted.performance.sharpe.toFixed(4)}</p>

                                    <p>Sortino Ratio: {optimizationResult.EquiWeighted.performance.sortino.toFixed(4)}</p>
                                    <p>Max Drawdown: {(optimizationResult.EquiWeighted.performance.max_drawdown * 100).toFixed(2)}%</p>
                                    <p>RoMaD: {optimizationResult.EquiWeighted.performance.romad.toFixed(4)}</p>

                                    <p>VaR 95%: {(optimizationResult.EquiWeighted.performance.var_95 * 100).toFixed(2)}%</p>
                                    <p>CVaR 95%: {(optimizationResult.EquiWeighted.performance.cvar_95 * 100).toFixed(2)}%</p>
                                    <p>VaR 90%: {(optimizationResult.EquiWeighted.performance.var_90 * 100).toFixed(2)}%</p>
                                    <p>CVaR 90%: {(optimizationResult.EquiWeighted.performance.cvar_90 * 100).toFixed(2)}%</p>
                                    <p>CAGR: {(optimizationResult.EquiWeighted.performance.cagr * 100).toFixed(2)}%</p>
                                    <p>Portfolio Beta: {optimizationResult.EquiWeighted.performance.portfolio_beta.toFixed(4)}</p>


                                    <h4 className="font-semibold mt-2">Weights:</h4>
                                    <ul>
                                        {Object.entries(optimizationResult.EquiWeighted.weights).map(([ticker, weight]) => (
                                            <li key={ticker}>
                                                {ticker}: {(weight * 100).toFixed(2)}%
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="plots-container">
                                    <ImageComponent
                                        base64String={optimizationResult.EquiWeighted.returns_dist}
                                        altText="EquiWeighted Distribution"
                                    />
                                    <ImageComponent
                                        base64String={optimizationResult.EquiWeighted.max_drawdown_plot}
                                        altText="EquiWeighted Drawdown"
                                    />
                                </div>
                            </div>
                        )}
                         {/* Critical Line Algorithm */}
                        {optimizationResult.CriticalLineAlgorithm && (
                            <div className="result-with-plot">
                                <div className="result-details">
                                    <h3 className="text-lg font-semibold">Critical Line Algorithm</h3>
                                    <p>Expected Return: {optimizationResult.CriticalLineAlgorithm.performance.expected_return.toFixed(4)}</p>
                                    <p>Volatility: {optimizationResult.CriticalLineAlgorithm.performance.volatility.toFixed(4)}</p>
                                    <p>Sharpe Ratio: {optimizationResult.CriticalLineAlgorithm.performance.sharpe.toFixed(4)}</p>

                                    <p>Sortino Ratio: {optimizationResult.CriticalLineAlgorithm.performance.sortino.toFixed(4)}</p>
                                    <p>Max Drawdown: {(optimizationResult.CriticalLineAlgorithm.performance.max_drawdown * 100).toFixed(2)}%</p>
                                    <p>RoMaD: {optimizationResult.CriticalLineAlgorithm.performance.romad.toFixed(4)}</p>

                                    <p>VaR 95%: {(optimizationResult.CriticalLineAlgorithm.performance.var_95 * 100).toFixed(2)}%</p>
                                    <p>CVaR 95%: {(optimizationResult.CriticalLineAlgorithm.performance.cvar_95 * 100).toFixed(2)}%</p>
                                    <p>VaR 90%: {(optimizationResult.CriticalLineAlgorithm.performance.var_90 * 100).toFixed(2)}%</p>
                                    <p>CVaR 90%: {(optimizationResult.CriticalLineAlgorithm.performance.cvar_90 * 100).toFixed(2)}%</p>
                                    <p>CAGR: {(optimizationResult.CriticalLineAlgorithm.performance.cagr * 100).toFixed(2)}%</p>

                                    <h4 className="font-semibold mt-2">Weights:</h4>
                                    <ul>
                                        {Object.entries(optimizationResult.CriticalLineAlgorithm.weights).map(([ticker, weight]) => (
                                            <li key={ticker}>
                                                {ticker}: {(weight * 100).toFixed(2)}%
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                <div className="plots-container">
                                    <ImageComponent
                                        base64String={optimizationResult.CriticalLineAlgorithm.returns_dist}
                                        altText="Critical Line Algorithm Distribution"
                                    />
                                    <ImageComponent
                                        base64String={optimizationResult.CriticalLineAlgorithm.max_drawdown_plot}
                                        altText="Critical Line Algorithm Drawdown"
                                    />
                                </div>
                            </div>
                        )}

                        {/* Cumulative Returns Chart */}
                        <div className="mt-6">
                            <h2 className="text-xl font-semibold mb-2">Cumulative Returns Over Time</h2>
                            <Line
                                data={prepareChartData(optimizationResult)}
                                options={chartOptions}
                            />
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
                    flex-direction: row;
                    gap: 20px;
                    align-items: flex-start;
                    margin-top: 20px;
                }

                .result-details {
                    flex: 1;
                }

                .plots-container {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }

                .image-container {
                    border: 1px solid #ccc;
                    padding: 10px;
                    background-color: #f9f9f9;
                }
            `}</style>
        </div>
    );
};

export default HomePage;
