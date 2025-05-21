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

const ModernPortfolioTheoryPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Modern Portfolio Theory (MPT) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Modern Portfolio Theory (MPT), a framework for constructing portfolios that maximize expected return for a given level of risk." />
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
            Modern Portfolio Theory (MPT)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Maximizing returns for a given level of risk through diversification
          </Typography>
        </Box>
        
        {/* Core Idea */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Core Idea
          </Typography>
          <Typography paragraph>
            Modern Portfolio Theory, introduced by <strong>Harry Markowitz</strong> in 1952, provides a <strong>mean-variance framework</strong> for building portfolios that <strong>maximize expected return for a chosen level of risk—or equivalently, minimize risk for a chosen return</strong>. Risk is proxied by the variance (or standard deviation) of returns, and the critical insight is that an asset's desirability depends not on its own risk-return profile but on <strong>how it co-moves with every other asset in the portfolio</strong>.
          </Typography>
        </Paper>
        
        {/* Key Assumptions */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Key Assumptions
          </Typography>
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                <strong>Risk-averse investors</strong>: prefer lower variance for equal expected return.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Returns are jointly distributed</strong> with finite means and variances; investors care only about those first two moments.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Single-period horizon</strong> with homogeneous expectations.
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <strong>Frictionless markets</strong> (no taxes, transaction costs, or limits on shorting).
              </Typography>
            </Box>
          </Box>
          <Typography paragraph sx={{ mt: 2 }}>
            These idealized conditions let the optimization collapse to simple quadratic programming.
          </Typography>
        </Paper>
        
        {/* Mathematical Model */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Mathematical Model
          </Typography>
          <Typography paragraph>
            Let
          </Typography>
          <Box component="ul" sx={{ pl: 4 }}>
            <Box component="li">
              <Typography>
                <InlineMath math="\mathbf{w}\in\mathbb{R}^N" />: portfolio weights s.t. <InlineMath math="\sum_i w_i = 1" />
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <InlineMath math="\boldsymbol{\mu}\in\mathbb{R}^N" />: expected returns
              </Typography>
            </Box>
            <Box component="li">
              <Typography>
                <InlineMath math="\boldsymbol{\Sigma}\in\mathbb{R}^{N\times N}" />: covariance matrix
              </Typography>
            </Box>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Risk–tolerance form
          </Typography>
          <Equation math="\min_{\mathbf{w}}\; \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w} \;-\; q\,\boldsymbol{\mu}^\top\mathbf{w} \qquad(q\ge0 \text{ is risk-tolerance})" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Target-return form
          </Typography>
          <Equation math="\begin{aligned} \min_{\mathbf{w}}\;& \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w} \\ \text{s.t. }& \boldsymbol{\mu}^\top\mathbf{w}= \mu^*, \quad \mathbf{1}^\top\mathbf{w}=1 \end{aligned}" />
          
          <Typography paragraph sx={{ mt: 2 }}>
            Solving either version generates one point on the <strong>efficient frontier</strong>—the set of portfolios delivering the <strong>highest return for each risk level</strong>.
          </Typography>
        </Paper>
        
        {/* Efficient Frontier & Capital Allocation Line */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Efficient Frontier & Capital Allocation Line
          </Typography>
          <Typography paragraph>
            Plotting expected return (vertical) against portfolio volatility (horizontal) yields a <strong>Markowitz bullet</strong>; its <strong>upper-left boundary</strong> is the efficient frontier. Introducing a risk-free asset spins a straight <strong>Capital Allocation Line (CAL)</strong> from the risk-free rate tangent to the frontier; its tangency portfolio maximizes the <strong>Sharpe ratio</strong> and underpins the Capital Asset Pricing Model (CAPM).
          </Typography>
          
          <Box sx={{ textAlign: 'center', my: 3 }}>
            {/* Placeholder for an image - in production, replace with actual image */}
            <Paper 
              elevation={0}
              sx={{ 
                width: '100%', 
                height: 300, 
                bgcolor: '#f0f0f0', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center' 
              }}
            >
              <Typography variant="body2" color="text.secondary">
                [Efficient Frontier and Capital Allocation Line Visualization]
              </Typography>
            </Paper>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              The efficient frontier (blue curve) and capital allocation line (red line) with the tangency portfolio
            </Typography>
          </Box>
        </Paper>
        
        {/* Diversification Benefit */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Diversification Benefit
          </Typography>
          <Typography paragraph>
            Because portfolio variance includes <strong>covariances</strong>, combining imperfectly correlated assets lowers overall volatility—allowing higher expected return for the same risk. This mathematical formalization of <strong>"don't put all your eggs in one basket"</strong> remains the most cited justification for global multi-asset investing.
          </Typography>
          
          <Typography paragraph>
            The power of diversification is one of the few "free lunches" in finance, allowing investors to reduce risk without necessarily sacrificing return. This principle is mathematically demonstrated in the portfolio variance formula:
          </Typography>
          
          <Equation math="\sigma_p^2 = \sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \sigma_i \sigma_j \rho_{ij}" />
          
          <Typography paragraph>
            When correlation (<InlineMath math="\rho_{ij}" />) between assets is less than 1, the portfolio variance is less than the weighted average of individual variances, creating the diversification benefit.
          </Typography>
        </Paper>
        
        {/* Criticisms & Extensions (Brief) */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Criticisms & Extensions
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Response / Extension</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Non-normal returns</strong> (skew, kurtosis)</TableCell>
                  <TableCell>Post-Modern Portfolio Theory; higher-moment optimizers</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Parameter uncertainty</strong></TableCell>
                  <TableCell>Bayesian & resampled MVO; robust optimization</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Single factor</strong></TableCell>
                  <TableCell>CAPM, then multi-factor (Fama–French, Carhart)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Estimation error in covariances</strong></TableCell>
                  <TableCell>Shrinkage estimators (Ledoit-Wolf)—used in our backend</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. (1952)</strong>. "Portfolio Selection." <em>The Journal of Finance</em>, 7(1), 77-91.
                <MuiLink href="https://doi.org/10.1111/j.1540-6261.1952.tb01525.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Markowitz, H. (1959)</strong>. <em>Portfolio Selection: Efficient Diversification of Investments</em>. John Wiley & Sons.
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
                  Mean-Variance Optimization
                </Typography>
                <Typography variant="body2" paragraph>
                  The cornerstone optimization technique of Modern Portfolio Theory that balances return and risk.
                </Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Capital Asset Pricing Model
                </Typography>
                <Typography variant="body2" paragraph>
                  The single-factor model derived from MPT that explains the relationship between systematic risk and expected return.
                </Typography>
                <Link href="/docs/capm" passHref>
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

export default ModernPortfolioTheoryPage; 