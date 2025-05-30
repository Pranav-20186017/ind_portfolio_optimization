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

const CAPMBetaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>CAPM Beta (β) for Indian Markets | QuantPort India Docs</title>
        <meta name="description" content="Understand CAPM Beta (β) for Indian equity portfolios. Learn how to measure systematic risk of NSE/BSE stocks and their sensitivity to the Indian market's movements." />
        <meta property="og:title" content="CAPM Beta (β) for Indian Markets | QuantPort India Docs" />
        <meta property="og:description" content="Understand CAPM Beta (β) for Indian equity portfolios. Learn how to measure systematic risk of NSE/BSE stocks and their sensitivity to the Indian market's movements." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/capm-beta" />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">
              ← Back to Docs
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
            CAPM Beta (β)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Measuring market-related risk in investments
          </Typography>
        </Box>
        
        {/* Introduction Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why Beta Matters
          </Typography>
          <Typography paragraph>
            <strong>Beta (β)</strong> is the cornerstone of the <Link href="/docs/capm" passHref><MuiLink>Capital Asset Pricing Model (CAPM)</MuiLink></Link>. 
            It quantifies how sensitive a security or portfolio is to broad-market movements. Beta is a key metric for understanding 
            investment risk and plays a crucial role in portfolio construction and risk management.
          </Typography>
          <Typography paragraph>
            Beta answers a simple yet essential question every investor asks:
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              "If the market rises (or falls) 1%, what do I expect my investment to do?"
            </Typography>
          </Box>
          
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%', bgcolor: '#f5f5f5' }}>
                <Typography variant="h6" gutterBottom align="center">
                  <InlineMath math="\beta \approx 1" />
                </Typography>
                <Typography align="center">
                  Moves in-line with the market
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%', bgcolor: '#fff3e0' }}>
                <Typography variant="h6" gutterBottom align="center">
                  <InlineMath math="\beta > 1" />
                </Typography>
                <Typography align="center">
                  Amplifies market swings (more volatile)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%', bgcolor: '#e8f5e9' }}>
                <Typography variant="h6" gutterBottom align="center">
                  <InlineMath math="\beta < 1" />
                </Typography>
                <Typography align="center">
                  Dampens market swings (less volatile)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%', bgcolor: '#e3f2fd' }}>
                <Typography variant="h6" gutterBottom align="center">
                  <InlineMath math="\beta < 0" />
                </Typography>
                <Typography align="center">
                  Tends to move opposite the market
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* The CAPM Equation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            The CAPM Equation
          </Typography>
          <Typography paragraph>
            Beta is derived from the fundamental CAPM regression equation:
          </Typography>
          <Equation math="\boxed{\; R_i - R_f \;=\; \alpha_i \;+\; \beta_i\,(R_m - R_f) \;+\; \varepsilon_i \;}" />
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Key Components
          </Typography>
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Symbol</strong></TableCell>
                  <TableCell><strong>Meaning</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>
                    <InlineMath math="R_i" />
                  </TableCell>
                  <TableCell>Return of asset/portfolio <em>i</em></TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="R_m" />
                  </TableCell>
                  <TableCell>Return of the market index</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="R_f" />
                  </TableCell>
                  <TableCell>Risk-free rate (e.g., T-bill yield or G-Sec yield)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\alpha_i" />
                  </TableCell>
                  <TableCell><strong>Jensen's Alpha</strong> – performance unexplained by the market</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\beta_i" />
                  </TableCell>
                  <TableCell><strong>CAPM Beta</strong> – slope coefficient measuring relative volatility</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\varepsilon_i" />
                  </TableCell>
                  <TableCell>Error term (idiosyncratic shocks)</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            We estimate <InlineMath math="\beta_i" /> (and <InlineMath math="\alpha_i" />) via <strong>Ordinary Least Squares (OLS)</strong> regression on <strong>excess returns</strong>:
          </Typography>
          <Typography sx={{ ml: 4 }}>
            <InlineMath math="y_t = (R_{i,t}-R_{f,t})" /> and <InlineMath math="x_t = (R_{m,t}-R_{f,t})" />
          </Typography>
        </Paper>
        
        {/* OLS Matrix Form */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            OLS in Matrix Form
          </Typography>
          <Typography paragraph>
            For the mathematically inclined, here's how beta is calculated using matrix notation:
          </Typography>
          
          <Typography paragraph>
            Let
          </Typography>
          <Equation math="\mathbf{y}\;=\;\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix},\quad
\mathbf{X}\;=\;\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix},
\quad\boldsymbol{\theta}\;=\;\begin{bmatrix}\alpha\\\beta\end{bmatrix}" />
          
          <Typography paragraph>
            OLS solves the minimization problem:
          </Typography>
          <Equation math="\min_{\boldsymbol{\theta}}\;(\mathbf{y}-\mathbf{X}\boldsymbol{\theta})^{\!\top}
(\mathbf{y}-\mathbf{X}\boldsymbol{\theta})" />
          
          <Typography paragraph>
            With the closed-form solution:
          </Typography>
          <Equation math="\boxed{\;
\hat{\boldsymbol{\theta}} = (\mathbf{X}^{\!\top}\mathbf{X})^{-1}\mathbf{X}^{\!\top}\mathbf{y}
\;}" />
          
          <Box sx={{ mt: 2, ml: 4 }}>
            <Typography>
              • <InlineMath math="\hat{\beta}" /> is the second element of <InlineMath math="\hat{\boldsymbol{\theta}}" />.
            </Typography>
            <Typography>
              • <InlineMath math="\hat{\alpha}" /> is the first element (excess return unexplained by the market).
            </Typography>
          </Box>
        </Paper>
        
        {/* Backend Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            How Our Portfolio Optimizer Computes Beta
          </Typography>
          <Typography paragraph>
            Inside our optimization backend, the following steps are performed to calculate beta:
          </Typography>
          
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2 }}>
            <ol>
              <li>
                <Typography paragraph>
                  <strong>Align dates</strong> of portfolio and benchmark.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Compute daily excess returns</strong>
                </Typography>
                <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto', fontSize: '0.875rem' }}>
                  <code>
                    port_excess  = port_returns - rf_series{'\n'}
                    bench_excess = benchmark_returns - rf_series
                  </code>
                </Box>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Run OLS regression</strong> (using <em>statsmodels</em>):
                </Typography>
                <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto', fontSize: '0.875rem' }}>
                  <code>
                    X = sm.add_constant(bench_excess.values)   # adds intercept{'\n'}
                    model   = sm.OLS(port_excess.values, X){'\n'}
                    result  = model.fit(){'\n'}
                    beta    = result.params[1]      # slope{'\n'}
                    alpha   = result.params[0] * 252  # annualise{'\n'}
                    p_value = result.pvalues[1]{'\n'}
                    r2      = result.rsquared
                  </code>
                </Box>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Fallback</strong> to the covariance method if OLS cannot run (for tiny samples).
                </Typography>
              </li>
            </ol>
          </Box>
          
          <Typography paragraph>
            The system also annualizes and caps extreme betas, then stores:
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <InlineMath math="\text{portfolio\_beta}" /> (<InlineMath math="\beta" />)
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\text{portfolio\_alpha}" /> (<InlineMath math="\alpha" />)
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <InlineMath math="\text{beta\_pvalue}" />, <InlineMath math="\text{r\_squared}" /> – statistical significance & fit quality.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Beta as Relative Volatility */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Beta as Relative Volatility
          </Typography>
          <Typography paragraph>
            Because <InlineMath math="\beta = \dfrac{\operatorname{Cov}(R_i,R_m)}{\operatorname{Var}(R_m)}" />:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Magnitude</strong> → <em>how volatile</em> the asset is relative to the market.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sign</strong> → <em>direction</em> of co-movement (negative if moving opposite).
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Practical Interpretation
          </Typography>
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Beta</strong></TableCell>
                  <TableCell><strong>Interpretation</strong></TableCell>
                  <TableCell><strong>Typical Suitability</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><InlineMath math="\beta \approx 0" /></TableCell>
                  <TableCell>Market-neutral</TableCell>
                  <TableCell>Market-neutral funds</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="0 < \beta < 1" /></TableCell>
                  <TableCell>Less volatile than market</TableCell>
                  <TableCell>Defensive stocks, utilities</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="\beta \approx 1" /></TableCell>
                  <TableCell>Market-like</TableCell>
                  <TableCell>Broad index funds</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="\beta > 1" /></TableCell>
                  <TableCell>More volatile</TableCell>
                  <TableCell>Growth / tech equities</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="\beta < 0" /></TableCell>
                  <TableCell>Inverse correlation</TableCell>
                  <TableCell>Hedging instruments</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Box sx={{ p: 2, mt: 3, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography paragraph>
              <strong>Example:</strong> A stock with a beta of 1.5 would be expected to rise by approximately 1.5% when the market rises by 1%, 
              and fall by approximately 1.5% when the market falls by 1%. Such high-beta stocks typically offer higher potential returns 
              during bull markets but also greater losses during bear markets.
            </Typography>
          </Box>
        </Paper>
        
        {/* Limitations & Good Practice */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages vs Limitations
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
                      <strong>Intuitive interpretation</strong> — Clear representation of how an asset co-moves with the market.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Simple calculation</strong> — Can be computed from readily available return data.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Theoretical foundation</strong> — Firmly grounded in modern portfolio theory and CAPM.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Risk assessment</strong> — Identifies systematic risk that cannot be diversified away.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Performance attribution</strong> — Helps distinguish between market-driven returns and alpha.
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
                      <strong>Non-stationarity</strong> — Beta values drift over time and are not constant.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Benchmark sensitivity</strong> — Results highly dependent on the market index chosen.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Historical bias</strong> — Past relationships may not predict future behavior.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Simplified model</strong> — Ignores other factors that affect returns (size, value, momentum).
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Assumes market efficiency</strong> — May not hold in markets with significant inefficiencies.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Good Practice */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Good Practices
          </Typography>
          
          <Typography paragraph>
            To address the limitations of beta, consider these best practices:
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Mitigation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Non-stationarity</strong> – β drifts over time</TableCell>
                  <TableCell>Compute <strong>rolling betas</strong> (our API does this yearly)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Leverage effects & heteroskedasticity</strong></TableCell>
                  <TableCell>Robust regressions or GARCH betas</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Choice of market proxy</strong></TableCell>
                  <TableCell>Use the most relevant benchmark (e.g., NIFTY 50 vs SENSEX)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Model simplicity</strong></TableCell>
                  <TableCell>Multifactor models (Fama-French, Carhart) capture size, value, momentum effects</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph sx={{ mt: 3 }}>
            Our portfolio optimizer addresses some of these concerns by calculating rolling betas to help you visualize how a portfolio's 
            relationship with the market may change over different time periods. This feature is especially valuable for long-term 
            investors who need to understand how their portfolio's risk characteristics evolve over time.
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
                <strong>Sharpe, W. F. (1964)</strong>. "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." <em>Journal of Finance</em>, 19(3), 425–442.
                <MuiLink href="https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1964.tb02865.x" target="_blank" sx={{ ml: 1 }}>
                  Access the paper
                </MuiLink>
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Lintner, J. (1965)</strong>. "Security Prices, Risk, and Maximal Gains from Diversification." <em>Journal of Finance</em>, 20(4), 587–615.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Black, F. (1972)</strong>. "Capital Market Equilibrium with Restricted Borrowing." <em>Journal of Business</em>, 45(3), 444–455.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Bodie, Z., Kane, A., & Marcus, A.</strong> <em>Investments</em> (12 ed.). McGraw-Hill, 2021 – Ch. 9 (CAPM & β estimation).
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
                  Capital Asset Pricing Model (CAPM)
                </Typography>
                <Typography variant="body2" paragraph>
                  The foundational theory behind beta, explaining the relationship between systematic risk and expected return.
                </Typography>
                <Link href="/docs/capm" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Blume-Adjusted Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A modified beta calculation that adjusts for the tendency of betas to revert toward the market average over time.
                </Typography>
                <Link href="/docs/blume-adjusted-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Jensen's Alpha (α)
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk-adjusted performance measure that represents the excess return of a portfolio over what CAPM predicts.
                </Typography>
                <Link href="/docs/jensens-alpha" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Rolling Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A time-series analysis of beta that shows how an asset's relationship with the market changes over different periods.
                </Typography>
                <Link href="/docs/rolling-beta" passHref>
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

export default CAPMBetaPage; 