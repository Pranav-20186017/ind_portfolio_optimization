type Stock = {
    ticker: string;
    name: string;
};

type Props = {
    selectedStocks: Stock[];
    onRemove: (ticker: string) => void;
};

const SelectedStocks = ({ selectedStocks, onRemove }: Props) => (
    <div className="mt-6 w-full">
        <h2 className="text-2xl font-semibold mb-4 text-center text-gray-700">
            Selected Stocks
        </h2>
        {selectedStocks.length > 0 ? (
            <div className="grid gap-3">
                {selectedStocks.map((stock) => (
                    <div
                        key={stock.ticker}
                        className="flex justify-between items-center border p-3 rounded-lg shadow-sm bg-gray-50"
                    >
                        <div className="text-gray-800 font-medium">
                            {stock.name} ({stock.ticker})
                        </div>
                        <button
                            onClick={() => onRemove(stock.ticker)}
                            className="text-red-500 text-xl font-bold hover:text-red-600 transition"
                        >
                            &times;
                        </button>
                    </div>
                ))}
            </div>
        ) : (
            <p className="text-gray-500 text-center">No stocks selected.</p>
        )}
    </div>
);

export default SelectedStocks;
