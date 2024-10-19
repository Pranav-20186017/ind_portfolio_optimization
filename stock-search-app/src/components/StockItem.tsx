type Props = {
    stock: {
        ticker: string;
        name: string;
    };
    onAdd: (ticker: string, name: string) => void;
};

const StockItem = ({ stock, onAdd }: Props) => (
    <div className="flex justify-between items-center border p-2 rounded">
        <div>
            {stock.name} ({stock.ticker})
        </div>
        <button
            onClick={() => onAdd(stock.ticker, stock.name)}
            className="bg-green-500 text-white px-2 py-1 rounded"
        >
            Add
        </button>
    </div>
);

export default StockItem;
