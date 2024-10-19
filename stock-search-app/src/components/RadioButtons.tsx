type Props = {
    exchange: 'BSE' | 'NSE' | 'NSE_SME';
    setExchange: (exchange: 'BSE' | 'NSE' | 'NSE_SME') => void;
};

const RadioButtons = ({ exchange, setExchange }: Props) => (
    <div className="flex gap-6 justify-center mb-6">
        {['BSE', 'NSE', 'NSE_SME'].map((ex) => (
            <label key={ex} className="flex items-center gap-2">
                <input
                    type="radio"
                    value={ex}
                    checked={exchange === ex}
                    onChange={() => setExchange(ex as 'BSE' | 'NSE' | 'NSE_SME')}
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-gray-700 text-lg font-medium">{ex}</span>
            </label>
        ))}
    </div>
);

export default RadioButtons;
