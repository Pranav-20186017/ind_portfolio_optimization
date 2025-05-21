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

const BlumeAdjustedBetaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Blume-Adjusted Beta | Portfolio Optimization</title>
        <meta name="description" content="Learn about Blume-Adjusted Beta, a modified beta calculation that adjusts for the tendency of betas to revert toward the market average over time." />
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
            Blume-Adjusted Beta (β)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A modified beta calculation that accounts for the mean-reversion tendency of systematic risk
          </Typography>
        </Box>
        
        {/* Why Adjust Beta */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Why Adjust Beta?
          </Typography>
          <Typography paragraph>
            Empirical studies show that <strong>historical CAPM betas drift toward 1.0 as you move forward in time</strong>—high betas fall, low betas rise.
            If you plug an <em>un-adjusted</em> β straight into cost-of-equity models, you can <strong>overstate</strong> risk for aggressive stocks and <strong>understate</strong> risk for defensive ones.
          </Typography>
          <Typography paragraph>
            To correct this "mean-reversion" bias, Marshall Blume (1971, <em>Journal of Finance</em>) proposed a simple linear transformation—now called the <strong>Blume Adjustment</strong>—that shrinks every beta part-way toward the market average of 1.0.
          </Typography>
        </Paper>
        
        {/* The Formula */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            The Formula
          </Typography>
          
          <Equation math="\boxed{\; \beta_{\text{Blume}} \;=\; 1 \;+\; b\,\bigl(\beta_{\text{raw}}-1\bigr) \;}" />
          
          <Typography component="div">
            <ul>
              <li>
                <Typography paragraph>
                  <InlineMath math="\beta_{\text{raw}}" /> = historical (OLS) beta you just estimated.
                </Typography>
              </li>
              <li>
                <Typography paragraph>
                  <InlineMath math="b" /> = shrinkage factor derived from long-run regressions. 
                  <em>Classic</em> Blume uses <InlineMath math="b = 0.67" />; sometimes literature quotes 2⁄3 or 0.60.
                </Typography>
              </li>
            </ul>
          </Typography>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Interpretation:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                High β (&gt; 1) is <strong>pulled down</strong>.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Low β (&lt; 1) is <strong>pushed up</strong>.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Market β (= 1) stays exactly 1.
              </Typography>
            </li>
          </ul>
        </Paper>
        
        {/* Derivation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Derivation in One Paragraph
          </Typography>
          
          <Typography paragraph>
            Blume regressed <strong>future betas (β<sub>t+5</sub>)</strong> on <strong>current betas (β<sub>t</sub>)</strong> for NYSE stocks:
          </Typography>
          
          <Equation math="\beta_{t+5} \;=\; a \;+\; b\,\beta_{t} \;+\;\varepsilon" />
          
          <Typography paragraph>
            He found <InlineMath math="a \approx 0.33,\; b \approx 0.67" />.
            Setting <InlineMath math="\beta_{t} = 1" /> implies <InlineMath math="\beta_{t+5} = 1" />—the market's average—hence the shrinkage interpretation.
          </Typography>
        </Paper>
        
        {/* Implementation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Our Implementation
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
              b = 0.67                       # Blume factor{'\n'}
              blume_adjusted_beta = 1 + b * (portfolio_beta - 1)
            </Typography>
          </Box>
          
          <Typography variant="h6" component="h3" gutterBottom sx={{ mt: 3 }}>
            Implementation Details
          </Typography>
          
          <ol>
            <li>
              <Typography paragraph>
                <strong>Raw Beta Calculation</strong>: First, we calculate the portfolio beta using ordinary least squares (OLS) regression of daily portfolio returns against benchmark returns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Blume Adjustment Factor</strong>: The adjustment uses the standard Blume factor of 0.67, which has been empirically validated across many markets and time periods.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Calculation Precision</strong>: The adjustment is applied with full floating-point precision to maintain accuracy in the final risk assessments.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Data Display</strong>: The result is displayed in the performance metrics of each optimization method, providing a forward-looking estimate of systematic risk.
              </Typography>
            </li>
          </ol>
          
          <Typography paragraph>
            This implementation ensures that beta values used in risk assessment and performance attribution represent a more reliable estimate of future systematic risk exposure, avoiding the pitfalls of using raw historical beta values directly.
          </Typography>
        </Paper>
        
        {/* Practical Usage */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Practical Usage
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Application</strong></TableCell>
                  <TableCell><strong>Why Blume Helps</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Cost of Equity / WACC</strong></TableCell>
                  <TableCell>Produces more stable, forward-looking betas, reducing forecast error in DCF models.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Risk budgeting</strong></TableCell>
                  <TableCell>Avoids systematically punishing high-β names in future projections.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Portfolio attribution</strong></TableCell>
                  <TableCell>Distinguishes <em>skill</em> (alpha) from temporary beta spikes that revert.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Limitations
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Issue</strong></TableCell>
                  <TableCell><strong>Mitigation / Note</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>One-size-fits-all b-factor</strong></TableCell>
                  <TableCell>Some analysts re-estimate <strong>b</strong> on their universe every few years.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Assumes linear mean-reversion</strong></TableCell>
                  <TableCell>Bayesian or Vasicek adjustments may fit better when distributions non-linear.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Short horizon portfolios</strong></TableCell>
                  <TableCell>If you rebalance monthly, reversion may be negligible—raw β could be enough.</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
        
        {/* Alternative Adjustments */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Alternative Adjustments
          </Typography>
          
          <TableContainer component={Paper} elevation={0}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>Method</strong></TableCell>
                  <TableCell><strong>Formula</strong></TableCell>
                  <TableCell><strong>Comment</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell><strong>Vasicek (Bayesian) Beta</strong></TableCell>
                  <TableCell>Shrink β toward mean by factor based on estimation variance.</TableCell>
                  <TableCell>Data-driven ("James–Stein") approach.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>Bloomberg Adjusted Beta</strong></TableCell>
                  <TableCell>β<sub>ʙʙ</sub> = 0.66·β_raw + 0.33·1</TableCell>
                  <TableCell>Same spirit as Blume but uses 2-year weekly returns.</TableCell>
                </TableRow>
                <TableRow>
                  <TableCell><strong>No adjustment</strong></TableCell>
                  <TableCell>β̂ = β_raw</TableCell>
                  <TableCell>Fine for intraday or very short-term trading.</TableCell>
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
                <strong>Blume, M. (1971).</strong> "On the Assessment of Risk." <em>Journal of Finance</em>, 26(1), 1–10.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Damodaran, A.</strong> <em>Investment Valuation</em>, 3 ed. Wiley, 2012 — Ch. 4 (Bottom-up Betas).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Bodie, Kane & Marcus.</strong> <em>Investments</em>, 12 ed. — Exhibit 13.2.
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
                  Treynor Ratio
                </Typography>
                <Typography variant="body2" paragraph>
                  A portfolio performance metric that measures returns earned in excess of the risk-free rate per unit of market risk (beta).
                </Typography>
                <Link href="/docs/treynor-ratio" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Conclusion */}
        <Paper elevation={2} sx={{ p: 4, mt: 4 }}>
          <Typography variant="body1">
            Blume-Adjusted Beta gives users a <strong>tempered, forward-looking view of market sensitivity</strong>, closing the gap between historical estimation and practical forecasting.
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

export default BlumeAdjustedBetaPage; 