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

// Reusable Equation component for consistent math rendering
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const RollingBetaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Rolling Beta (β) | Portfolio Optimization</title>
        <meta name="description" content="Learn about Rolling Beta (β), a dynamic measure of systematic risk that tracks how an asset's sensitivity to the market evolves through time." />
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
            Rolling Beta (β)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Tracking the evolution of systematic risk over time
          </Typography>
        </Box>
        
        {/* Section 1: What Is Rolling Beta */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What Is Rolling Beta?
          </Typography>
          <Typography paragraph>
            A <strong>Rolling Beta</strong> tracks how an asset's <strong>systematic risk</strong>—its sensitivity to the market—<strong>evolves through time</strong>.
            Instead of estimating a single, "static" CAPM β from the full sample, we:
          </Typography>
          <Box sx={{ pl: 3, mb: 2 }}>
            <ol>
              <li>
                <Typography paragraph>
                  <strong>Slice</strong> the return history into overlapping (or calendar) windows.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Estimate β</strong> separately inside each window (usually via an OLS regression of excess returns on market excess returns).
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Stitch</strong> those estimates together to create a time-series of β values.
                </Typography>
              </li>
            </ol>
          </Box>
          <Typography paragraph>
            The result is a curve that answers:
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mb: 2, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              "Was this stock defensive in 2020 but high-beta in 2021?"
            </Typography>
          </Box>
        </Paper>
        
        {/* Section 2: Why Investors Care */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why Investors Care
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Insight</strong></TableCell>
                  <TableCell><strong>Practical Action</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Regime shifts</strong> – β drifts upward in bull runs, collapses in crises.</TableCell>
                  <TableCell>Adjust hedging or leverage dynamically.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Style changes</strong> – a fund once market-neutral may become directional.</TableCell>
                  <TableCell>Performance attribution & manager monitoring.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Stress testing</strong> – identify periods when β spiked {'>'} 1.5 (higher draw-down risk).</TableCell>
                  <TableCell>Scenario analysis & risk-budget alerts.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          <Typography paragraph>
            Rolling β is therefore a key input for <strong>tactical asset allocation</strong>, <strong>risk budgeting</strong>, and <strong>manager due-diligence</strong>.
          </Typography>
        </Paper>
        
        {/* Section 3: Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Implementation at a Glance
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom>
            Window Choice
          </Typography>
          <Box sx={{ pl: 3, mb: 3 }}>
            <ul>
              <li>
                <Typography paragraph>
                  <strong>Window length:</strong> <strong>1 calendar year</strong> (≈ 252 trading days).
                  <br/>
                  <em>Rationale:</em> strikes a balance between statistical power and responsiveness.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Frequency:</strong> <strong>Daily returns.</strong>
                </Typography>
              </li>
            </ul>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom>
            3.2 Code Path
          </Typography>
          <Box 
            component="pre" 
            sx={{ 
              p: 2, 
              bgcolor: '#f5f5f5', 
              borderRadius: 1, 
              overflow: 'auto',
              fontFamily: 'monospace',
              fontSize: '0.875rem'
            }}
          >
{`# srv.py  → compute_yearly_betas()
df = pd.DataFrame({"p": port_excess, "b": bench_excess})
beta_y = df.groupby(df.index.year).apply(
    lambda g: g["p"].cov(g["b"]) / g["b"].var()
)`}
          </Box>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            <em>Highlights</em>
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ol>
              <li>
                <Typography paragraph>
                  Converts daily <strong>excess returns</strong> to a common calendar year.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  Uses the <strong>covariance / variance</strong> formula (mathematically identical to the OLS slope).
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  Returns a <code>{'{year: β}'}</code> dictionary, which your frontend plots as the <strong>Rolling Betas</strong> line chart.
                </Typography>
              </li>
            </ol>
          </Box>
          <Typography paragraph variant="body2" sx={{ fontStyle: 'italic' }}>
            (Your results page overlays the "market β = 1" line for quick context.)
          </Typography>
        </Paper>
        
        {/* Section 4: Math Walk-Through */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Math Walk-Through
          </Typography>
          <Typography paragraph>
            For a given window $t=1,\dots,T$:
          </Typography>
          
          <Typography variant="h6" gutterBottom>
            1. <strong>Excess returns</strong>
          </Typography>
          <Equation math="y_t = R_{i,t} \;-\; R_{f,t},\quad x_t = R_{m,t} \;-\; R_{f,t}" />
          
          <Typography variant="h6" gutterBottom>
            2. <strong>OLS (slope only)</strong>
          </Typography>
          <Equation math="\hat{\beta}_w = 
     \frac{\displaystyle\sum_{t=1}^{T}
             (x_t-\bar{x})(y_t-\bar{y})}
          {\displaystyle\sum_{t=1}^{T}
             (x_t-\bar{x})^2}
   \;=\;
   \frac{\operatorname{Cov}(x,y)}
        {\operatorname{Var}(x)}" />
          
          <Typography variant="h6" gutterBottom>
            3. <strong>Repeat</strong> for each subsequent window → produce a sequence
          </Typography>
          <Equation math="\bigl\{\hat{\beta}_{\text{1995}},
          \hat{\beta}_{\text{1996}},\dots\bigr\}" />
          
          <Typography variant="h6" gutterBottom>
            4. <strong>Interpret</strong> spikes, drops, and trends against economic events.
          </Typography>
          
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, mt: 3, fontStyle: 'italic', borderLeft: '4px solid #2196f3' }}>
            <Typography>
              <strong>Tip:</strong> overlapping windows (e.g., 252-day rolling with daily step) give smoother curves but require more computation. You chose calendar-year buckets—excellent for quick visual stories.
            </Typography>
          </Box>
        </Paper>
        
        {/* Section 5: Interpreting the Rolling-β Chart */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting the Rolling-β Chart
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Pattern</strong></TableCell>
                  <TableCell><strong>Meaning</strong></TableCell>
                  <TableCell><strong>Possible Action</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Steady β ≈ 1</strong></TableCell>
                  <TableCell>Market-like behaviour</TableCell>
                  <TableCell>Hold as core allocation</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>β trending ↑</strong></TableCell>
                  <TableCell>Becoming more cyclical / growth-tilted</TableCell>
                  <TableCell>Reduce weight or hedge</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>β {'<'} 0</strong> episode</TableCell>
                  <TableCell>Acts as market hedge (rare)</TableCell>
                  <TableCell>Exploit for diversification</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Erratic β</strong></TableCell>
                  <TableCell>Unstable factor exposures</TableCell>
                  <TableCell>Review strategy, increase monitoring</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          <Typography paragraph>
            Always cross-check with <strong>p-values / R²</strong> if sample size is small.
          </Typography>
        </Paper>
        
        {/* Section 6: Best Practices & Caveats */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Best Practices & Caveats
          </Typography>
          <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Recommendation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Window length trade-off</strong></TableCell>
                  <TableCell>Short → noisy; Long → lags. Test 6-m vs 1-y vs 3-y.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Benchmark choice</strong></TableCell>
                  <TableCell>Use the <strong>same</strong> index your investors benchmark against (NIFTY 50, S&P 500…).</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Non-stationarity</strong></TableCell>
                  <TableCell>Combine rolling β with <strong>time-series break tests</strong> to alert when behaviour changes statistically.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Leverage / derivatives</strong></TableCell>
                  <TableCell>Un‐dampened β can explode if the portfolio carries leverage. Scale exposure before regression.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Section 7: Extending the Analysis */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Extending the Analysis
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ol>
              <li>
                <Typography paragraph>
                  <strong>Rolling <InlineMath math="\alpha" /> and <InlineMath math="R^2" /></strong> – parallel time-series reveal skill drift and explanatory power.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Downside Beta</strong> – compute β using <em>only</em> market down-days for crash sensitivity.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Multi-factor rolling</strong> – size, value, momentum betas over time (uses Fama-French regressions).
                </Typography>
              </li>
            </ol>
          </Box>
        </Paper>
        
        {/* Section 8: Key References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Key References
          </Typography>
          <Box sx={{ pl: 3 }}>
            <ol>
              <li>
                <Typography paragraph>
                  <strong>Ferson, W. E., & Schadt, R. W. (1996)</strong> – <em>Measuring Fund Strategy and Performance in Changing Economic Conditions.</em> <strong>Journal of Finance</strong>, 51(2), 425-461.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Blitz, D. (2013)</strong> – <em>Benchmarking Low-Volatility Strategies.</em> <strong>Journal of Portfolio Management</strong>, 40(2), 89-100.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <strong>Bodie, Z., Kane, A., & Marcus, A.</strong> – <em>Investments</em> (12 ed.), Ch. 24: Performance Attribution.
                </Typography>
              </li>
            </ol>
          </Box>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Brooks, R. D., Faff, R. W., & McKenzie, M. D. (1998)</strong>. "Time-varying beta risk of Australian industry portfolios: A comparison of modelling techniques." <em>Australian Journal of Management</em>, 23(1), 1-22.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Mergner, S., & Bulla, J. (2008)</strong>. "Time-varying beta risk of Pan-European industry portfolios: A comparison of alternative modeling techniques." <em>The European Journal of Finance</em>, 14(8), 771-802.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Fama, E. F., & French, K. R. (1992)</strong>. "The cross-section of expected stock returns." <em>The Journal of Finance</em>, 47(2), 427-465.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Alexander, C. (2008)</strong>. <em>Market Risk Analysis, Volume II: Practical Financial Econometrics</em>. John Wiley & Sons.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Conclusion */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography paragraph sx={{ fontStyle: 'italic' }}>
            Rolling Betas turn a single risk number into a <strong>movie</strong>—revealing how your strategy's market sensitivity morphs through cycles. Paired with your backend's automated yearly computation and interactive frontend chart, users get an immediate, intuitive feel for regime changes and hidden risks.
          </Typography>
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
                  CAPM Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  Understand the foundation of static beta calculation before exploring its time-varying properties.
                </Typography>
                <Link href="/docs/capm-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Jensen's Alpha
                </Typography>
                <Typography variant="body2" paragraph>
                  Combine with rolling beta to track both market sensitivity and excess return generation over time.
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
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  Monitor how risk-adjusted performance changes as beta evolves through different market regimes.
                </Typography>
                <Link href="/docs/sharpe-ratio" passHref>
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

export default RollingBetaPage; 