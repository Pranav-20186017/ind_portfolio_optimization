// src/pages/_app.tsx

import '../styles/globals.css';
import type { AppProps } from 'next/app';
import { Analytics } from '@vercel/analytics/react'; // Import Vercel Analytics
import { SpeedInsights } from "@vercel/speed-insights/next"
function MyApp({ Component, pageProps }: AppProps) {
    return (
        <>
            <Component {...pageProps} />
            <Analytics /> {/* Add Vercel Analytics */}
             <SpeedInsights /> {}
        </>
    );
}

export default MyApp;
