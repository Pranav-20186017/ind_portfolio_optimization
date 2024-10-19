import { useState, useEffect } from 'react';
import SearchBar from '@/components/SearchBar';
import SelectedStocks from '@/components/SelectedStocks';
import RadioButtons from '@/components/RadioButtons';
import axios from 'axios';

type StockData = {
    [key: string]: {
        [key: string]: {
            name: string;
            isin: string;
        };
    };
};

type Stock = {
    ticker: string;
    name: string;
};

export default function Home() {
    const [searchTerm, setSearchTerm] = useState<string>('');
    const [filteredStocks, setFilteredStocks] = useState<Stock[]>([]);
    const [selectedStocks, setSelectedStocks] = useState<Stock[]>([]);
    const [exchange, setExchange] = useState<'BSE' | 'NSE' | 'NSE_SME'>('BSE');
    const [stockData, setStockData] = useState<StockData>({ BSE: {}, NSE: {}, NSE_SME: {} });

    useEffect(() => {
        fetch('/stock_data.json')
            .then((res) => res.json())
            .then((data) => setStockData(data))
            .catch((err) => console.error('Error loading stock data:', err));
    }, []);

    useEffect(() => {
        if (searchTerm.trim()) {
            const stocks = Object.keys(stockData[exchange])
                .filter((ticker) =>
                    stockData[exchange][ticker].name.toLowerCase().includes(searchTerm.toLowerCase())
                )
                .map((ticker) => ({
                    ticker,
                    name: stockData[exchange][ticker].name,
                }));

            setFilteredStocks(stocks.slice(0, 5)); // Limit to 5 suggestions for better performance
        } else {
            setFilteredStocks([]);
        }
    }, [searchTerm, exchange, stockData]);

    const addStock = (stock: Stock) => {
        if (!selectedStocks.some((s) => s.ticker === stock.ticker)) {
            setSelectedStocks([...selectedStocks, stock]);
        }
        setSearchTerm('');
        setFilteredStocks([]);
    };

    const removeStock = (ticker: string) => {
        setSelectedStocks(selectedStocks.filter((stock) => stock.ticker !== ticker));
    };

    const resetPage = () => {
        setSearchTerm('');
        setFilteredStocks([]);
        setSelectedStocks([]);
        setExchange('BSE');
    };

    const handleSubmit = async () => {
        try {
            const response = await axios.post('/api/submit', { selectedStocks });
            console.log('Submitted successfully:', response.data);
        } catch (error) {
            console.error('Error submitting data:', error);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-100 to-gray-300 p-4">
            <div className="bg-white shadow-lg rounded-2xl p-8 w-full max-w-lg flex flex-col gap-6">
                <h1 className="text-4xl font-bold text-center text-indigo-700 mb-4">
                    Stock Search App
                </h1>

                <RadioButtons exchange={exchange} setExchange={setExchange} />

                <SearchBar searchTerm={searchTerm} setSearchTerm={setSearchTerm} />

                {filteredStocks.length > 0 && (
                    <div className="border border-gray-300 rounded-lg shadow-lg max-h-40 overflow-y-auto bg-white mb-4 animate-dropdown transition-all duration-300 ease-out">
                        {filteredStocks.map((stock) => (
                            <div
                                key={stock.ticker}
                                onClick={() => addStock(stock)}
                                className="cursor-pointer p-3 hover:bg-indigo-100 transition-colors duration-200 delay-75"
                            >
                                {stock.name} ({stock.ticker})
                            </div>
                        ))}
                    </div>
                )}

                <SelectedStocks selectedStocks={selectedStocks} onRemove={removeStock} />

                <div className="flex gap-4 mt-6 justify-center">
                    <button
                        onClick={handleSubmit}
                        className="px-6 py-3 bg-indigo-600 text-white rounded-full shadow-md hover:bg-indigo-700 transition-all duration-200"
                    >
                        Submit
                    </button>
                    <button
                        onClick={resetPage}
                        className="px-6 py-3 bg-red-600 text-white rounded-full shadow-md hover:bg-red-700 transition-all duration-200"
                    >
                        Reset
                    </button>
                </div>
            </div>
        </div>
    );
}
