// src/pages/_app.tsx

import '../styles/globals.css';
import type { AppProps } from 'next/app';
import Head from 'next/head';
import { Analytics } from '@vercel/analytics/react'; // Import Vercel Analytics
import { SpeedInsights } from "@vercel/speed-insights/next"

function MyApp({ Component, pageProps }: AppProps) {
    const noindex = process.env.NODE_ENV !== 'production';
    
    return (
        <>
            <Head>
                {noindex && <meta name="robots" content="noindex,nofollow" />}
            </Head>
            <Component {...pageProps} />
            <Analytics /> {/* Add Vercel Analytics */}
            <SpeedInsights />
        </>
    );
}

export default MyApp;
