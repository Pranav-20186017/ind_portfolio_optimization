// src/pages/index.tsx

import React, { useState, useEffect } from 'react';
import { StockData, StockOption } from '../types';
import Autocomplete, { createFilterOptions } from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';

const HomePage: React.FC = () => {
    const [stockData, setStockData] = useState<StockData>({});
    const [options, setOptions] = useState<StockOption[]>([]);
    const [searchValue, setSearchValue] = useState('');
    const [selectedStocks, setSelectedStocks] = useState<StockOption[]>([]);
    const [value, setValue] = useState<StockOption | null>(null); // Added state to control the value

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
        setValue(null); // Clear the selected value
        setSearchValue(''); // Clear the search input
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

    const handleSubmit = () => {
        // Group selected stocks by exchange
        const groupedStocks = selectedStocks.reduce(
            (acc: Record<string, string[]>, stock) => {
                acc[stock.exchange] = acc[stock.exchange] || [];
                acc[stock.exchange].push(stock.ticker);
                return acc;
            },
            {}
        );

        // Send data to API (console.log for now)
        for (const [exchange, tickers] of Object.entries(groupedStocks)) {
            const dataToSend = {
                tickers,
                exchange,
            };
            console.log('Sending data:', dataToSend);
            // Replace the console.log with actual API call when ready
        }
    };

    const handleReset = () => {
        setSelectedStocks([]);
        setSearchValue('');
        setValue(null); // Ensure the value is reset
    };

    return (
        <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Stock Search</h1>
            <Autocomplete
                options={options}
                getOptionLabel={(option) =>
                    `${option.ticker} - ${option.name} (${option.exchange})`
                }
                value={value} // Control the value
                onChange={handleAddStock}
                inputValue={searchValue}
                onInputChange={(event, newInputValue) => {
                    setSearchValue(newInputValue);
                }}
                renderInput={(params) => (
                    <TextField {...params} label="Search Stock" variant="outlined" />
                )}
                style={{ width: '100%', maxWidth: 600 }}
                openOnFocus={false} // Prevent dropdown from opening on focus
                filterOptions={(opts, params) => {
                    const filtered = createFilterOptions<StockOption>()(opts, params);
                    // Show options only when input has some value
                    if (params.inputValue === '') {
                        return [];
                    }
                    return filtered;
                }}
                clearOnBlur={false} // Keep input focused after selection
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
        </div>
    );
};

export default HomePage;
