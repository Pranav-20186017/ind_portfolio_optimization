import StockItem from './StockItem';

type Stock = {
    ticker: string;
    name: string;
};

type Props = {
    stocks: Stock[];
    onAdd: (ticker: string, name: string) => void;
};

const StockList = ({ stocks, onAdd }: Props) => (
    <div className="grid gap-2">
        {stocks.length > 0 ? (
            stocks.map((stock) => (
                <StockItem key={stock.ticker} stock={stock} onAdd={onAdd} />
            ))
        ) : (
            <p className="text-gray-500">No stocks found.</p>
        )}
    </div>
);

export default StockList;
