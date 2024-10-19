import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
    if (req.method === 'POST') {
        const { selectedStocks } = req.body;
        console.log('Received selected stocks:', selectedStocks);
        res.status(200).json({ message: 'Stocks submitted successfully!' });
    } else {
        res.setHeader('Allow', ['POST']);
        res.status(405).end(`Method ${req.method} Not Allowed`);
    }
}
