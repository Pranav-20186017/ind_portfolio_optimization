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

const VolatilityPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Volatility | Portfolio Optimization</title>
        <meta name="description" content="Learn about Volatility, a measure of how widely returns disperse around their average." />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
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
            Volatility (σ)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A measure of how widely returns disperse around their average
          </Typography>
        </Box>
        
        {/* What Volatility Means */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What Volatility Means
          </Typography>
          <Typography paragraph>
            Volatility measures <strong>how widely returns disperse around their average</strong>.
            In practical terms it tells you, "How bumpy is the ride?"—the greater the volatility, the larger the typical up- or down-swings you can expect over the period in question.
          </Typography>
        </Paper>
        
        {/* Mathematical Definition */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Definition
          </Typography>
          <Typography paragraph>
            For a series of <InlineMath math="n" /> returns <InlineMath math="r_1,\,r_2,\dots ,r_n" />:
          </Typography>
          
          <Equation math="\sigma \;=\; \sqrt{\frac{1}{n-1}\sum_{t=1}^{n}\bigl(r_t-\bar{r}\bigr)^2} \qquad \bigl(\bar{r}= \tfrac1n\sum r_t\bigr)" />
          
          <ul>
            <li>
              <Typography paragraph>
                <strong>Units</strong> → the same as returns (e.g., % per day).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Annualisation</strong> → <InlineMath math="\sigma_{\text{annual}} = \sigma_{\text{daily}}\times\sqrt{252}" /> (for trading-day data).
              </Typography>
            </li>
          </ul>
          
          <Typography paragraph>
            Because the formula squares deviations, it penalises big moves heavily, capturing both upside and downside swings.
          </Typography>
        </Paper>
        
        {/* How It's Computed */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Implementation
          </Typography>
          <Typography paragraph>
            Inside our implementation, you'll find:
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
              volatility = port_returns.std() * np.sqrt(252)
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Key points
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Aspect</strong></TableCell>
                  <TableCell><strong>Detail</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Data frequency</TableCell>
                  <TableCell><strong>Daily</strong> simple returns (<code>port_returns</code>)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Estimator</TableCell>
                  <TableCell>Sample standard deviation (<code>Series.std()</code> uses <InlineMath math="n-1" /> denominator)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Annual factor</TableCell>
                  <TableCell>√252 multiplies daily σ to yearly</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Storage</TableCell>
                  <TableCell>Result sits in <code>performance.volatility</code> for each optimisation method (MVO, MinVol, HRP, …)</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Interpreting Volatility */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting Volatility
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Annual σ</strong></TableCell>
                  <TableCell><strong>Typical Asset Class</strong></TableCell>
                  <TableCell><strong>Rule-of-thumb Daily Move (±1 σ)</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>3 %–5 %</strong></TableCell>
                  <TableCell>3-month T-bill</TableCell>
                  <TableCell>0.20 %–0.30 bps</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>10 %</strong></TableCell>
                  <TableCell>Investment-grade bonds</TableCell>
                  <TableCell>~0.6 %</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>15 %</strong></TableCell>
                  <TableCell>Large-cap equities</TableCell>
                  <TableCell>~1.0 %</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>30 %+</strong></TableCell>
                  <TableCell>Crypto, single tech stocks</TableCell>
                  <TableCell>1.9 % +</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph sx={{ mt: 2 }}>
            <em>Higher σ ⇒ wider return range; not inherently "bad," but must be balanced against expected return (Sharpe, Sortino, Treynor).</em>
          </Typography>
        </Paper>
        
        {/* Why Investors Track σ */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why Investors Track σ
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Position sizing</strong> – allocate less capital to high-σ components.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk limits</strong> – trading desks impose daily VaR or σ ceilings.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sharpe & Sortino</strong> – denominator uses σ (or downside σ).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio optimisation</strong> – MPT minimises σ for a target return.
              </Typography>
            </li>
          </ol>
        </Paper>
        
        {/* Mini FAQ */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mini FAQ
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Is volatility the same as risk?
          </Typography>
          <Typography paragraph sx={{ ml: 4 }}>
            It's a <em>proxy</em> for price uncertainty; it doesn't distinguish upside vs. downside. Complement with drawdown and tail metrics.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Can σ be negative?
          </Typography>
          <Typography paragraph sx={{ ml: 4 }}>
            No—by definition it's a square-root of squared deviations.
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            How big is "big"?
          </Typography>
          <Typography paragraph sx={{ ml: 4 }}>
            Compare to peer assets or benchmarks; 20 % annual σ for equities is typical, 60 % is extreme.
          </Typography>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Hull, J.</strong> <em>Options, Futures, and Other Derivatives</em>, 11 ed. – Ch. 1 (Volatility Basics).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Pafka & Kondor (2003)</strong> – <em>Estimated correlation matrices…</em> Physica A 319.
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
                  Expected Returns
                </Typography>
                <Typography variant="body2" paragraph>
                  The weighted-average outcome you anticipate earning on an asset or portfolio over a stated horizon.
                </Typography>
                <Link href="/docs/expected-returns" passHref>
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
                <Link href="/docs/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Value-at-Risk
                </Typography>
                <Typography variant="body2" paragraph>
                  A statistical measure that quantifies the level of financial risk within a portfolio over a specific time frame.
                </Typography>
                <Link href="/docs/value-at-risk" passHref>
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

export default VolatilityPage; 