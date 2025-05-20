import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Divider
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const MinCDaRPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Minimum Conditional Drawdown at Risk (MinCDaR) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Minimum Conditional Drawdown at Risk (MinCDaR) portfolio optimization, an approach that focuses on minimizing portfolio drawdowns and recovery periods." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/education" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Education
            </Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">
              ← Back to Portfolio Optimizer
            </Button>
          </Link>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Minimum Conditional Drawdown at Risk (MinCDaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Portfolio optimization focusing on minimizing drawdown severity
          </Typography>
        </Box>
        
        {/* Placeholder content - to be updated later */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Content Coming Soon
          </Typography>
          <Typography paragraph>
            Detailed information about Minimum Conditional Drawdown at Risk (MinCDaR) portfolio optimization will be added here soon.
          </Typography>
        </Paper>
      </Container>
    </>
  );
};

// Add getStaticProps to generate this page at build time
export const getStaticProps = async () => {
  return {
    props: {},
  };
};

export default MinCDaRPage; 