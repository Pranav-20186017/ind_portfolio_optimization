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
  TableRow
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

const ExpectedReturnsPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Expected Return | Portfolio Optimization</title>
        <meta name="description" content="Learn about Expected Return, the weighted-average outcome you anticipate earning on an asset or portfolio over a stated horizon." />
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
            Expected Return
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            The weighted-average outcome you anticipate earning on an asset or portfolio
          </Typography>
        </Box>
        
        {/* Concept in One Sentence */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Concept in One Sentence
          </Typography>
          <Typography paragraph>
            <strong>Expected return</strong> is the weighted-average outcome you <em>anticipate</em> earning on an asset or portfolio over a stated horizon—usually expressed as an annual percentage. It is the "gravity-center" of the distribution of possible future returns.
          </Typography>
        </Paper>
        
        {/* Formal Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Formal Definition
          </Typography>
          <Typography paragraph>
            For a discrete distribution of <InlineMath math="k" /> possible returns <InlineMath math="r_1,\dots ,r_k" /> with probabilities <InlineMath math="p_1,\dots ,p_k" />:
          </Typography>
          
          <Equation math="\mathbb{E}[R]=\sum_{j=1}^{k} p_j\,r_j" />
          
          <Typography paragraph>
            For a continuous distribution <InlineMath math="f(r)" />:
          </Typography>
          
          <Equation math="\mathbb{E}[R]=\int_{-\infty}^{\infty} r\,f(r)\,dr" />
          
          <Typography paragraph>
            In practice we rarely know the true distribution, so we <strong>estimate</strong> <InlineMath math="\mathbb{E}[R]" /> from historical data or a forward-looking model.
          </Typography>
        </Paper>
        
        {/* Estimation Techniques */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Estimation Techniques
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Method</strong></TableCell>
                  <TableCell><strong>Core Idea</strong></TableCell>
                  <TableCell><strong>Strengths</strong></TableCell>
                  <TableCell><strong>Drawbacks</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Historical Mean</strong></TableCell>
                  <TableCell>Average of past returns: <InlineMath math="\hat{\mu}=\frac{1}{n}\sum_{t=1}^{n}r_t" />.</TableCell>
                  <TableCell>Simple, transparent, data-driven.</TableCell>
                  <TableCell>Sensitive to sample window, ignores regime changes.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Exponentially-Weighted Mean</strong></TableCell>
                  <TableCell>Recent returns get higher weight.</TableCell>
                  <TableCell>Reacts to trends.</TableCell>
                  <TableCell>Still backward-looking; picks decay factor heuristically.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>CAPM Implied Return</strong></TableCell>
                  <TableCell><InlineMath math="R_f + \beta(\mathbb{E}[R_m]-R_f)" />.</TableCell>
                  <TableCell>Links return to priced market risk.</TableCell>
                  <TableCell>Relies on stable β and market premium.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Black–Litterman</strong></TableCell>
                  <TableCell>Blends market-implied returns with subjective views.</TableCell>
                  <TableCell>Controls estimation error; consistent with equilibrium.</TableCell>
                  <TableCell>Requires equilibrium priors & subjective confidence.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Fundamental Factor Models</strong></TableCell>
                  <TableCell>Expected return = sum of factor exposures × premia.</TableCell>
                  <TableCell>Incorporates earnings growth, value, momentum, etc.</TableCell>
                  <TableCell>Needs robust factor premia forecasts.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* How Your Backend Computes It */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Implementation
          </Typography>
          <Typography paragraph>
            Inside our implementation, PyPortfolioOpt's helper is called:
          </Typography>
          
          <Box sx={{ 
            p: 2, 
            bgcolor: 'rgba(0, 0, 0, 0.04)', 
            borderRadius: 1, 
            fontFamily: 'monospace',
            overflowX: 'auto',
            my: 2
          }}>
            <Typography variant="body2">
              from pypfopt import expected_returns<br />
              mu = expected_returns.mean_historical_return(df, frequency=252)
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Details
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Data</strong>: price level DataFrame <code>df</code> (already adjusted for splits & dividends).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Frequency</strong>: <code>252</code> → converts daily mean to <em>annualised</em> arithmetic mean:
              </Typography>
              <Equation math="\hat{\mu}_{\text{annual}} = \bar{r}_{\text{daily}} \times 252" />
            </li>
            <li>
              <Typography paragraph>
                <strong>Result</strong>: vector <InlineMath math="\boldsymbol{\mu}" /> feeds directly into the Efficient Frontier optimiser and is also cached for reporting.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Interpreting Expected Return in Your Results Cards */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting Expected Return in Results
          </Typography>
          <Typography paragraph>
            When viewing optimization results, you'll see the <strong>Expected Return</strong> (expressed as an annual percentage) displayed alongside <strong>Volatility</strong> and <strong>Sharpe Ratio</strong>. Keep these important points in mind:
          </Typography>
          
          <ul>
            <li>
              <Typography paragraph>
                Expected return is a <strong>statistical estimate</strong>, not a guarantee—it represents the center of the probability distribution of possible outcomes.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                When comparing different optimization methods, focus on <em>risk-adjusted</em> metrics (like Sharpe or Sortino ratios) rather than maximizing expected return alone, as higher returns typically come with increased risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                The expected return value serves as a key input for calculating various performance metrics (such as Treynor Ratio and Information Ratio) that help evaluate portfolio efficiency.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H.</strong> "Portfolio Selection." <em>Journal of Finance</em> (1952).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Black, F. & Litterman, R.</strong> "Global Portfolio Optimization." <em>Financial Analysts Journal</em> (1992).
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Efficient Frontier
                </Typography>
                <Typography variant="body2" paragraph>
                  The set of optimal portfolios that offer the highest expected return for a defined level of risk.
                </Typography>
                <Link href="/education/efficient-frontier" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Modern Portfolio Theory
                </Typography>
                <Typography variant="body2" paragraph>
                  A framework for constructing portfolios that maximize expected return for a given level of market risk.
                </Typography>
                <Link href="/education/modern-portfolio-theory" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of risk-adjusted return that helps investors understand the return of an investment compared to its risk.
                </Typography>
                <Link href="/education/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  CAPM Beta (β)
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of systematic risk that represents how an asset moves relative to the overall market.
                </Typography>
                <Link href="/education/capm-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
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

export default ExpectedReturnsPage; 