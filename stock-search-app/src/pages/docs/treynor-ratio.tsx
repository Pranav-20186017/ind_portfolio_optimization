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

const TreynorRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Treynor Ratio for Indian Portfolios | QuantPort India Docs</title>
        <meta name="description" content="Evaluate Indian stock portfolios with the Treynor Ratio. Measure excess returns per unit of systematic risk for NSE/BSE securities to optimize market-related risk exposure." />
        <meta property="og:title" content="Treynor Ratio for Indian Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Evaluate Indian stock portfolios with the Treynor Ratio. Measure excess returns per unit of systematic risk for NSE/BSE securities to optimize market-related risk exposure." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/treynor-ratio" />
      </Head>
      
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation Buttons */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Button variant="outlined" color="primary" component={Link} href="/docs">
            ← Back to Docs
          </Button>
          <Button variant="outlined" color="secondary" component={Link} href="/">
            ← Back to Portfolio Optimizer
          </Button>
        </Box>
        
        {/* Title Section */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Treynor Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Measuring excess return per unit of systematic risk
          </Typography>
        </Box>
        
        {/* What the Treynor Ratio Measures */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            What the Treynor Ratio Measures
          </Typography>
          <Typography paragraph>
            <strong>Treynor Ratio</strong> (invented by Jack Treynor, 1965) gauges <em>how much excess return</em> a portfolio delivers <strong>per unit of systematic risk</strong> (<MuiLink component={Link} href="/docs/capm-beta">β</MuiLink>).
            It answers the question:
          </Typography>
          <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, borderLeft: '4px solid #3f51b5' }}>
            <Typography variant="body1" component="blockquote" sx={{ fontStyle: 'italic' }}>
              "For each percentage point of market‐related risk I bear, how much am I paid above the risk-free rate?"
            </Typography>
          </Box>
        </Paper>
        
        {/* Formula Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Formula
          </Typography>
          
          <Equation math="\boxed{\;T = \dfrac{R_p - R_f}{\beta_p}\;}" />
          
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
                    <InlineMath math="R_p" />
                  </TableCell>
                  <TableCell><strong>Annualised</strong> portfolio return</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="R_f" />
                  </TableCell>
                  <TableCell>Risk-free rate (T-bill / repo)</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>
                    <InlineMath math="\beta_p" />
                  </TableCell>
                  <TableCell>Portfolio <MuiLink component={Link} href="/docs/capm-beta">CAPM beta</MuiLink> (relative volatility to market)</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            <strong>Numerator</strong> → reward: excess return above risk-free.
          </Typography>
          <Typography paragraph>
            <strong>Denominator</strong> → risk: only non-diversifiable (systematic) risk.
          </Typography>
          
          <Box sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 1, borderLeft: '4px solid #3f51b5', my: 2 }}>
            <Typography paragraph>
              Contrast with <MuiLink component={Link} href="/docs/sharpe-ratio">Sharpe Ratio</MuiLink>, which divides by <strong>total</strong> volatility (<InlineMath math="\sigma" />).
              Treynor fits best when portfolio is well-diversified and unsystematic risk ≈ 0.
            </Typography>
          </Box>
        </Paper>
        
        {/* CAPM Link Section */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            CAPM Link
          </Typography>
          <Typography paragraph>
            From <MuiLink component={Link} href="/docs/capm">CAPM (Capital Asset Pricing Model)</MuiLink>:
          </Typography>
          
          <Equation math="\mathbb{E}[R_p] - R_f = \beta_p \bigl(\mathbb{E}[R_m] - R_f\bigr)" />
          
          <Typography paragraph>
            If the portfolio lies <em>on</em> the Security-Market Line, its Treynor Ratio equals the market risk premium:
          </Typography>
          
          <Equation math="T_{\text{CAPM}} = \mathbb{E}[R_m]-R_f" />
          
          <Typography paragraph>
            A <strong>higher</strong> <InlineMath math="T" /> implies positive <MuiLink component={Link} href="/docs/jensens-alpha">Jensen's Alpha</MuiLink>; <strong>lower</strong> implies under-performance.
          </Typography>
        </Paper>
        
        {/* Backend Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Computation in Our Backend
          </Typography>
          <Typography paragraph>
            1. <strong>Annual returns / β</strong> are already produced in <code>compute_custom_metrics</code>.
          </Typography>
          <Typography paragraph>
            2. Treynor is stored as:
          </Typography>
          
          <Box component="pre" sx={{ p: 2, bgcolor: '#f1f1f1', borderRadius: 1, overflow: 'auto', fontSize: '0.875rem' }}>
            <code>
              treynor_ratio = annual_excess / portfolio_beta   # beta from OLS; excess = R_p - R_f
            </code>
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <ul>
              <li>
                <Typography component="div">
                  Annual excess return uses 252-day factor.
                </Typography>
              </li>
              <li>
                <Typography component="div">
                  <MuiLink component={Link} href="/docs/capm-beta">β</MuiLink> comes from daily OLS regression against the chosen benchmark.
                </Typography>
              </li>
            </ul>
          </Box>
        </Paper>
        
        {/* Interpreting Values */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Interpreting Values
          </Typography>
          
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Treynor Ratio</strong></TableCell>
                  <TableCell><strong>Interpretation (given same benchmark)</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>&gt; MRP</strong></TableCell>
                  <TableCell>Out-performed market on a β-adjusted basis</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>≈ MRP</strong></TableCell>
                  <TableCell>In-line with <MuiLink component={Link} href="/docs/capm">CAPM</MuiLink> expectations</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>&lt; MRP</strong></TableCell>
                  <TableCell>Under-performed for its level of market risk</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Typography paragraph>
            <em>MRP = market risk premium = <InlineMath math="R_m-R_f" />.</em>
          </Typography>
        </Paper>
        
        {/* Strengths & Limitations */}
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
                      <strong>Ignores diversifiable risk</strong> — Ideal for well-diversified funds and assessing systematic risk exposure.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Comparable across funds</strong> — Directly compare funds with different total volatility but similar beta exposure.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Theoretical foundation</strong> — Aligns with CAPM theory and Security Market Line concepts.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Asset class assessment</strong> — More appropriate for evaluating portfolios within a specific asset class.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Manager skill insight</strong> — Provides clear view of a manager's ability to generate excess returns per unit of systematic risk.
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
                      <strong>Ignores idiosyncratic risk</strong> — Misleading if portfolio holds large non-systematic risk components.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Beta estimation sensitivity</strong> — Results highly dependent on beta calculation period and benchmark choice.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Risk asymmetry blindness</strong> — Ignores downside vs. upside risk differences that Sharpe & Sortino may capture.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Linear relationship assumption</strong> — Beta calculation assumes linear market relationship that may break during extreme conditions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Backward-looking</strong> — Historical beta may not be representative of future systematic risk exposure.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Not ideal for standalone evaluation</strong> — Ignores total risk which matters to undiversified investors.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="subtitle1" gutterBottom>
              <strong>When to prefer Treynor Ratio:</strong>
            </Typography>
            <Typography paragraph>
              1. Evaluating managers within the same market segment
            </Typography>
            <Typography paragraph>
              2. Comparing funds that are components of a broader diversified portfolio
            </Typography>
            <Typography paragraph>
              3. When systematic risk is the primary concern for the investor
            </Typography>
          </Box>
        </Paper>
        
        {/* Practical Use-Cases */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Practical Use-Cases
          </Typography>
          
          <ol>
            <li>
              <Typography component="div">
                <strong>Mutual-fund league tables</strong> — rank diversified equity funds.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Beta-target sleeves</strong> — choose the most efficient manager within a β band (e.g., 0.8–1.2).
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Risk budgeting</strong> — find strategies that maximise reward per unit of systematic risk rather than per unit of total volatility.
              </Typography>
            </li>
          </ol>
        </Paper>
        
        {/* Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Example
          </Typography>
          
          <TableContainer component={Paper} elevation={0} sx={{ mb: 3, maxWidth: '300px' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Metric</strong></TableCell>
                  <TableCell><strong>Fund A</strong></TableCell>
                  <TableCell><strong>Fund B</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Return</TableCell>
                  <TableCell>12 %</TableCell>
                  <TableCell>14 %</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>β</TableCell>
                  <TableCell>0.8</TableCell>
                  <TableCell>1.3</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><InlineMath math="R_f" /></TableCell>
                  <TableCell>5 %</TableCell>
                  <TableCell>5 %</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
          
          <Equation math="T_A=\frac{0.12-0.05}{0.8}=0.0875\quad T_B=\frac{0.14-0.05}{1.3}=0.0692" />
          
          <Typography paragraph>
            Even though Fund B earns higher raw return, <strong>Fund A</strong> delivers <strong>more reward per unit of β-risk</strong>.
          </Typography>
        </Paper>
        
        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            References
          </Typography>
          <ul>
            <li>
              <Typography component="div">
                <strong>Treynor, J. L. (1965)</strong>. "How to Rate Management of Investment Funds." <em>Harvard Business Review</em>, 43(1), 63-75.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Sharpe, W. F. (1966)</strong>. "Mutual Fund Performance." <em>The Journal of Business</em>, 39(1), 119-138.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Jensen, M. C. (1968)</strong>. "The Performance of Mutual Funds in the Period 1945-1964." <em>Journal of Finance</em>, 23(2), 389-416.
              </Typography>
            </li>
            <li>
              <Typography component="div">
                <strong>Bodie, Z., Kane, A., & Marcus, A. J.</strong> <em>Investments</em> (12 ed.). McGraw-Hill, 2021 – Ch. 24 (Portfolio Performance Evaluation).
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
                  CAPM Beta (β)
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of systematic risk that represents how an asset moves relative to the overall market.
                </Typography>
                <Button variant="contained" color="primary" component={Link} href="/docs/capm-beta">
                  Learn More
                </Button>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Jensen's Alpha (α)
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk-adjusted performance measure that represents the average return on a portfolio above or below CAPM predictions.
                </Typography>
                <Button variant="contained" color="primary" component={Link} href="/docs/jensens-alpha">
                  Learn More
                </Button>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Sharpe Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of risk-adjusted return that helps investors understand the return of an investment compared to its total risk.
                </Typography>
                <Button variant="contained" color="primary" component={Link} href="/docs/sharpe-ratio">
                  Learn More
                </Button>
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

export default TreynorRatioPage;
