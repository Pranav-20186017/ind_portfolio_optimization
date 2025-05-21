import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Re‑usable equation wrapper for consistent styling
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const EquallyWeightedPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Equally Weighted Portfolio | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn how the naïve 1/N diversification rule constructs a simple yet powerful Equally Weighted portfolio—complete with intuition, full math, and implementation details."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Equally Weighted Portfolio (1/N)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            The elegant power of naïve diversification
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            The <strong>Equally Weighted (EW)</strong> strategy assigns identical capital weights to every asset in the basket—
            a simple rule often written as <InlineMath math="w_i = 1 / N" /> for <InlineMath math="i = 1,\dots,N" /> assets.
            Despite its apparent naïveté, a long literature—from <em>Sharpe</em> (1964) to <em>DeMiguel, Garlappi & Uppal</em> (2009)—shows that EW
            performs surprisingly well out‑of‑sample because it sidesteps the estimation noise that plagues more sophisticated
            optimizers.
          </Typography>
          <Typography paragraph>
            In our optimiser this method serves both as a <em>robust baseline</em> and a practical choice for users who favour
            transparency, ultra‑low turnover, and broad diversification without making any forecast about returns or covariances.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Think of splitting a pizza with friends. If everyone gets the same‑sized slice, no negotiation is needed and nobody
            feels short‑changed. Likewise, an equally weighted portfolio gives each asset an identical "slice" of capital,
            ensuring the portfolio's outcome is not dominated by any single name.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Every‑ingredient salad analogy:</strong> When you mix equal spoonfuls of many ingredients, the flavour of any
              one component is muted. Similarly, EW dilution keeps idiosyncratic shocks from any stock from overwhelming the
              total portfolio.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            Given <InlineMath math="N" /> assets with expected return vector <InlineMath math="\mu \in \mathbb{R}^N" /> and covariance matrix
            <InlineMath math="\Sigma \in \mathbb{R}^{N\times N}" />, Equally Weighted sets weight vector
            <InlineMath math="w^{\text{EW}} = \tfrac{1}{N}\mathbf{1}" />, where <InlineMath math="\mathbf{1}" /> is a vector of ones.
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Expected Portfolio Return</strong></Typography>
            <Equation math="\mathbb{E}[R_p] = w^{\top}\mu = \frac{1}{N}\sum_{i=1}^{N} \mu_i" />
            <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}><strong>Portfolio Variance</strong></Typography>
            <Equation math="\sigma_p^2 = w^{\top}\Sigma w = \frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}\sigma_{ij}" />
          </Box>

          <Typography paragraph>
            If all pairwise correlations are zero and each asset shares the same variance <InlineMath math="\sigma^2" />, the variance simplifies to
            <InlineMath math="\sigma_p^2 = \sigma^2 / N" />. Thus <em>risk falls at the familiar 1/√N rate</em>—the core diversification benefit.
            In reality, correlations are positive so the decay is slower, but the intuition remains intact.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            The backend class <code>EquiWeightedOptimizer</code> in <code>srv.py</code> creates the weight vector in one line:
          </Typography>
          <Equation math="w_i = \frac{1}{N}\;\;\forall\,i" />
          <Typography paragraph>
            Performance metrics (Sharpe, Sortino, CAPM beta, etc.) are then computed exactly the same way as other methods,
            letting you compare EW head‑to‑head with MVO, HRP, and more.
          </Typography>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Suppose we invest in four stocks—A, B, C, D—each with different expected returns and volatilities. EW allocates 25 % to each.
            The portfolio's expected return is the simple average of individual returns, while variance is computed via the
            earlier formula. Even if stock D is twice as volatile as A, its impact is capped by the equal stake.
          </Typography>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages and Limitations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Forecast-free robustness:</strong> No estimation of expected returns or covariances required, eliminating a major source of error in portfolio construction.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Low turnover:</strong> Weights only drift as prices move; periodic rebalancing keeps transaction costs minimal compared to optimization-based approaches.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Small-cap tilt:</strong> Relative to market-cap weighted portfolios, equally weighted naturally emphasizes smaller names—often harvesting size premia.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Benchmark baseline:</strong> Provides a clean yard-stick to test whether more complex strategies add value in out-of-sample performance.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Simplicity and transparency:</strong> Easy to explain and implement, with no "black box" elements that might raise concerns for stakeholders or clients.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Limitations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Risk blindness:</strong> Ignores that some assets are far more volatile than others, potentially including excessively risky assets with the same weight as stable ones.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Liquidity constraints:</strong> A 1/N schedule may overweight illiquid micro-caps in large portfolios, creating practical implementation challenges.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>No covariance exploitation:</strong> Cannot achieve the lowest possible risk for a given return when reliable estimates exist, leaving diversification benefits on the table.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Concentration during bubbles:</strong> Without rebalancing, bubble assets can grow to dominate the portfolio as they inflate, leading to unintended concentration.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Sector imbalances:</strong> May lead to unintended sector or factor tilts depending on the initial universe of securities.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph><strong>Sharpe, W. F. (1964)</strong>. "Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk." <em>Journal of Finance</em>, 19(3), 425–442.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>DeMiguel, V., Garlappi, L., & Uppal, R. (2009)</strong>. "Optimal Versus Naïve Diversification: How Inefficient Is the 1/N Portfolio Strategy?" <em>Review of Financial Studies</em>, 22(5), 1915–1953.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Benartzi, S. & Thaler, R. (2001)</strong>. "Naïve Diversification Strategies in Defined Contribution Saving Plans." <em>American Economic Review</em>, 91(1), 79–98.</Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Topics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Mean‑Variance Optimization</Typography>
                <Typography variant="body2" paragraph>The classical risk–return optimiser balancing variance and expected return.</Typography>
                <Link href="/docs/mvo" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Minimum Volatility</Typography>
                <Typography variant="body2" paragraph>Optimiser that targets the absolute lowest portfolio risk.</Typography>
                <Link href="/docs/min-vol" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Hierarchical Risk Parity (HRP)</Typography>
                <Typography variant="body2" paragraph>A modern risk‑based allocation that clusters assets instead of inverting covariance.</Typography>
                <Link href="/docs/hrp" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Static generation hook (Next.js)
export const getStaticProps = async () => {
  return { props: {} };
};

export default EquallyWeightedPage;
