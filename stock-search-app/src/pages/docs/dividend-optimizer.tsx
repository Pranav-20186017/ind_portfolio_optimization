import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const DividendOptimizerDocsPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Dividend Optimizer (Forward-Yield + Entropy) | QuantPort India Docs</title>
        <meta name="description" content="Docs for the Dividend Optimizer: forward dividend yield with risk caps and greedy/MILP share allocation, plus entropy-regularized legacy variant." />
        <meta property="og:title" content="Dividend Optimizer (Forward-Yield + Entropy) | QuantPort India Docs" />
        <meta property="og:description" content="Docs for the Dividend Optimizer: forward dividend yield with risk caps and greedy/MILP share allocation, plus entropy-regularized legacy variant." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/dividend-optimizer" />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/dividend" passHref>
            <Button variant="outlined" color="secondary">← Back to Dividend App</Button>
          </Link>
        </Box>

        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Dividend Optimizer — Forward Yield and Entropy
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Two complementary optimizers for dividend portfolios: a forward-yield share allocator and an entropy-regularized continuous-weight variant
          </Typography>
        </Box>

        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Overview
          </Typography>
          <Typography paragraph>
            This page documents two dividend-focused optimizers available in the backend:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Forward-Yield Allocator (share-based)</strong>: builds a discrete-share portfolio to maximize expected cash yield subject to risk and allocation caps. It uses raw <em>Close</em> prices (no auto-adjust) and TTM dividends to estimate forward yield per security.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Entropy-regularized Optimizer (continuous weights)</strong>: maximizes a weighted combination of portfolio dividend yield and portfolio entropy to promote diversification, with an optional volatility cap.
              </Typography>
              <Equation math={"H(\\mathbf{w}) = -\\sum_i w_i \\log w_i"} />
            </li>
          </ul>
        </Paper>

        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Parameters
          </Typography>
          <Typography paragraph>
            Forward yields are computed from <strong>TTM (trailing-twelve-month)</strong> dividends over a rolling window and the most recent raw <em>Close</em> prices.
          </Typography>
          <Box component="div" sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
            <Typography component="div">• <strong>stocks</strong>: list of selections, each with <code>ticker</code> and <code>exchange</code> (NSE/BSE) to identify the security.</Typography>
            <Typography component="div">• <strong>budget</strong>: optional INR amount; when provided, outputs integer share allocation and cash deployment stats.</Typography>
            <Typography component="div">• <strong>entropy_weight</strong> (λ): strength of diversification via entropy in the objective (default 0.05).</Typography>
            <Typography component="div">• <strong>price_lookback_days</strong>: days of price history used to estimate covariance for risk constraints.</Typography>
            <Typography component="div">• <strong>yield_lookback_days</strong>: window length (days) for aggregating TTM dividends used in forward-yield estimation.</Typography>
            <Typography component="div">• <strong>min_weight_floor</strong> (ε): lower bound on each weight to ensure feasibility and avoid zero allocations.</Typography>
            <Typography component="div">• <strong>vol_cap</strong>: optional annualized volatility limit applied through <InlineMath math={"\\mathbf{w}^T \\Sigma \\mathbf{w}"} />.</Typography>
            <Typography component="div">• <strong>use_median_ttm</strong>: if true, smooths TTM by taking the 90-day median of the rolling TTM series for stability.</Typography>
          </Box>
        </Paper>

        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Explanation: Forward-Yield and Share Allocation
          </Typography>
          <Typography paragraph>
            For each ticker <InlineMath math="i" />, compute trailing-twelve-month dividends over a rolling window and last close price <InlineMath math="P_i" /> to estimate forward yield:
          </Typography>
          <Equation math={"y_i = \\dfrac{\\mathrm{TTM}_i}{P_i}"} />
          <Typography paragraph>
            Portfolio yield with weights <InlineMath math={"\\mathbf{w}"} /> is:
          </Typography>
          <Equation math={"y_p = \\sum_i w_i y_i"} />
          <Typography paragraph>
            Here <InlineMath math={"y_i"} /> denotes the individual forward yield of asset <InlineMath math={"i"} />, and <InlineMath math={"y_p"} /> denotes the portfolio forward yield.
          </Typography>
          <Typography paragraph>
            With a risk cap (optional):
          </Typography>
          <Equation math={"\\mathbf{w}^T \\Sigma \\mathbf{w} \\le \\sigma^2_{max}, \\quad \\sum_i w_i = 1, \\quad w_i \\ge \\epsilon"} />
        </Paper>

        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Entropy-Regularized Variant
          </Typography>
          <Typography paragraph>
            Objective (maximize expected dividend yield plus diversification):
          </Typography>
          <Equation math={"\\max_{\\mathbf{w} \\ge 0} \\; \\mathbf{y}^T \\mathbf{w} + \\lambda H(\\mathbf{w})"} />
          <Typography paragraph>Subject to:</Typography>
          <Box component="div" sx={{ ml: 2 }}>
            <Typography component="div"><InlineMath math={"\\sum_i w_i = 1"} /> (weights sum to one)</Typography>
            <Typography component="div"><InlineMath math={"w_i \\ge \\epsilon"} /> (long-only with a minimum floor)</Typography>
            <Typography component="div"><InlineMath math={"\\mathbf{w}^T \\Sigma \\mathbf{w} \\le \\sigma^2_{max}"} /> (optional annualized volatility cap)</Typography>
          </Box>
          <Typography paragraph sx={{ mt: 2 }}>
            Here <InlineMath math={"H(\\mathbf{w}) = -\\sum_i w_i \\log w_i"} /> and <InlineMath math={"\\Sigma"} /> is the covariance of daily returns scaled by 252.
          </Typography>
        </Paper>

        {/* Practical notes section removed per request */}

        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Entropy (Portfolio Diversity)
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of diversification used in the optimizer objective.
                </Typography>
                <Link href="/docs/entropy" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  CVaR & CDaR
                </Typography>
                <Typography variant="body2" paragraph>
                  Tail and drawdown risk measures that can complement income portfolios.
                </Typography>
                <Link href="/docs/min-cvar" passHref>
                  <Button variant="contained" color="primary">Min-CVaR</Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

export const getStaticProps = async () => {
  return { props: {} };
};

export default DividendOptimizerDocsPage;


