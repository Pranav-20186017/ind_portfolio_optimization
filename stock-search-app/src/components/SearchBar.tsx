type Props = {
    searchTerm: string;
    setSearchTerm: (term: string) => void;
};

const SearchBar = ({ searchTerm, setSearchTerm }: Props) => (
    <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search for stocks..."
        className="border border-gray-300 p-4 rounded-full w-full mb-6 focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-sm"
    />
);

export default SearchBar;
