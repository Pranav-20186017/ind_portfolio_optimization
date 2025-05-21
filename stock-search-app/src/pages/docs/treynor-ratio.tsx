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
        <title>Treynor Ratio | Portfolio Optimization</title>
        <meta name="description" content="Learn about the Treynor Ratio, a performance metric that measures excess return per unit of systematic risk (beta)." />
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
            <strong>Treynor Ratio</strong> (invented by Jack Treynor, 1965) gauges <em>how much excess return</em> a portfolio delivers <strong>per unit of systematic risk</strong> (<Link href="/docs/capm-beta" passHref><MuiLink>β</MuiLink></Link>).
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
                  <TableCell>Portfolio <Link href="/docs/capm-beta" passHref><MuiLink>CAPM beta</MuiLink></Link> (relative volatility to market)</TableCell>
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
              Contrast with <Link href="/docs/sharpe-ratio" passHref><MuiLink>Sharpe Ratio</MuiLink></Link>, which divides by <strong>total</strong> volatility (<InlineMath math="\sigma" />).
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
            From <Link href="/docs/capm" passHref><MuiLink>CAPM (Capital Asset Pricing Model)</MuiLink></Link>:
          </Typography>
          
          <Equation math="\mathbb{E}[R_p] - R_f = \beta_p \bigl(\mathbb{E}[R_m] - R_f\bigr)" />
          
          <Typography paragraph>
            If the portfolio lies <em>on</em> the Security-Market Line, its Treynor Ratio equals the market risk premium:
          </Typography>
          
          <Equation math="T_{\text{CAPM}} = \mathbb{E}[R_m]-R_f" />
          
          <Typography paragraph>
            A <strong>higher</strong> <InlineMath math="T" /> implies positive <Link href="/docs/jensens-alpha" passHref><MuiLink>Jensen's Alpha</MuiLink></Link>; <strong>lower</strong> implies under-performance.
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
                  <Link href="/docs/capm-beta" passHref><MuiLink>β</MuiLink></Link> comes from daily OLS regression against the chosen benchmark.
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
                  <TableCell>In-line with <Link href="/docs/capm" passHref><MuiLink>CAPM</MuiLink></Link> expectations</TableCell>
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
            Strengths & Limitations
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Strength</strong></TableCell>
                  <TableCell><strong>Limitation</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Ignores diversifiable risk—ideal for well-diversified funds.</TableCell>
                  <TableCell>Misleading if portfolio holds large idiosyncratic risk.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Directly comparable across funds with different σ but similar β.</TableCell>
                  <TableCell>Sensitive to β estimate instability; needs reliable benchmark.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Aligns with CAPM theory and SML.</TableCell>
                  <TableCell>Ignores downside vs. upside; Sharpe & Sortino may capture that nuance.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
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
                  Jensen's Alpha (α)
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk-adjusted performance measure that represents the average return on a portfolio above or below CAPM predictions.
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
                  A measure of risk-adjusted return that helps investors understand the return of an investment compared to its total risk.
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

export default TreynorRatioPage;
