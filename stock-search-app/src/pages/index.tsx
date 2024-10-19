// src/pages/index.tsx

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
import VirtualizedListbox from '../components/VirtualizedListbox';

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

    // Initialize Fuse.js
    const fuse = useMemo(() => {
        return new Fuse(options, {
            keys: ['ticker', 'name'],
            threshold: 0.3, // Adjust for desired fuzziness
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
            setFilteredOptions(results.slice(0, 50).map((result) => result.item)); // Limit to top 50 results
        }, 300),
        [fuse]
    );

    useEffect(() => {
        debouncedFilter(inputValue);
    }, [inputValue, debouncedFilter]);

    const handleAddStock = (event: any, newValue: StockOption | null) => {
        if (newValue) {
            // Avoid duplicates
            const isDuplicate = selectedStocks.some(
                (stock) =>
                    stock.ticker === newValue.ticker && stock.exchange === newValue.exchange
            );
            if (!isDuplicate) {
                setSelectedStocks([...selectedStocks, newValue]);
            }
        }
        setInputValue(''); // Clear the search input
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
        // Prepare the data to send to the backend
        const dataToSend = {
            stocks: selectedStocks.map((stock) => ({
                ticker: stock.ticker,
                exchange: stock.exchange,
            })),
        };

        // Send data to API
        try {
            const response = await axios.post('http://127.0.0.1:8000/optimize/', dataToSend);

            // Update the state with optimization result
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

    return (
        <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Stock Search</h1>
            <Autocomplete
                options={filteredOptions}
                getOptionLabel={(option) =>
                    `${option.ticker} - ${option.name} (${option.exchange})`
                }
                onChange={handleAddStock}
                inputValue={inputValue}
                onInputChange={(event, newInputValue) => {
                    setInputValue(newInputValue);
                }}
                renderInput={(params) => (
                    <TextField {...params} label="Search Stock" variant="outlined" />
                )}
                style={{ width: '100%', maxWidth: 600 }}
                openOnFocus={false} // Prevent dropdown from opening on focus
                filterOptions={(options) => options} // Disable default filtering
                clearOnBlur={false}
                value={null} // Clear input after selection
                ListboxComponent={VirtualizedListbox as React.ComponentType<React.HTMLAttributes<HTMLElement>>}
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
                >
                    Submit
                </Button>
                <Button variant="outlined" color="secondary" onClick={handleReset}>
                    Reset
                </Button>
            </div>
            <div className="mt-6">
                {optimizationResult && (
                    <div>
                        <h2 className="text-xl font-semibold mb-2">Optimization Results</h2>
                        {optimizationResult.MVO && (
                            <div>
                                <h3 className="text-lg font-semibold">
                                    Mean-Variance Optimization (MVO)
                                </h3>
                                <p>
                                    Expected Return:{' '}
                                    {optimizationResult.MVO.performance.expected_return.toFixed(4)}
                                </p>
                                <p>
                                    Volatility: {optimizationResult.MVO.performance.volatility.toFixed(4)}
                                </p>
                                <p>
                                    Sharpe Ratio: {optimizationResult.MVO.performance.sharpe.toFixed(4)}
                                </p>
                                <h4 className="font-semibold">Weights:</h4>
                                <ul>
                                    {Object.entries(optimizationResult.MVO.weights).map(([ticker, weight]) => (
                                        <li key={ticker}>
                                            {ticker}: {(weight * 100).toFixed(2)}%
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                        {optimizationResult.MinVol && (
                            <div>
                                <h3 className="text-lg font-semibold">
                                    Minimum Volatility Portfolio
                                </h3>
                                <p>
                                    Expected Return:{' '}
                                    {optimizationResult.MinVol.performance.expected_return.toFixed(4)}
                                </p>
                                <p>
                                    Volatility: {optimizationResult.MinVol.performance.volatility.toFixed(4)}
                                </p>
                                <p>
                                    Sharpe Ratio: {optimizationResult.MinVol.performance.sharpe.toFixed(4)}
                                </p>
                                <h4 className="font-semibold">Weights:</h4>
                                <ul>
                                    {Object.entries(optimizationResult.MinVol.weights).map(([ticker, weight]) => (
                                        <li key={ticker}>
                                            {ticker}: {(weight * 100).toFixed(2)}%
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default HomePage;