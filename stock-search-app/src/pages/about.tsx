import React from 'react';
import SEO from '../components/SEO';
import TopNav from '../components/TopNav';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Divider from '@mui/material/Divider';

export async function getStaticProps() {
  return { props: {}, revalidate: 86400 }; // daily
}

export default function About() {
  return (
    <>
      <SEO
        title="About – Indian Portfolio Optimization"
        description="What this app does, models used, data sources, and credits."
        path="/about"
        image="/og/default.png"
      />
      <TopNav />
      
      <Box sx={{ 
        maxWidth: 900, 
        mx: 'auto', 
        my: 5, 
        px: { xs: 2, md: 4 },
        py: { xs: 2, md: 0 } 
      }}>
        <Paper 
          elevation={0}
          sx={{ 
            p: { xs: 3, md: 5 }, 
            borderRadius: 2,
            border: '1px solid #f0f0f0',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)'
          }}
        >
          <Typography 
            component="h1" 
            variant="h4" 
            sx={{ 
              fontWeight: 700, 
              mb: 3,
              color: '#1e293b',
              fontSize: { xs: '1.75rem', md: '2rem' } 
            }}
          >
            About QuantPort India
          </Typography>
          
          <Divider sx={{ mb: 4, width: '80px', height: '3px', bgcolor: '#0052cc', borderRadius: '2px' }} />
          
          <Typography 
            variant="body1" 
            sx={{ 
              mb: 3,
              fontSize: '1.05rem',
              lineHeight: 1.7
            }}
          >
            <strong>QuantPort India</strong> is India's <strong>one-stop shop for stock portfolio optimization</strong>—trusted by both <strong>retail investors</strong> and <strong>professional fund managers</strong>. Our AI-powered platform combines robust quantitative finance with practical usability, providing state-of-the-art portfolio construction and risk management tools tailored for Indian equities.
          </Typography>
          
          <Typography 
            variant="body1" 
            sx={{ 
              mb: 4,
              fontSize: '1.05rem',
              lineHeight: 1.7
            }}
          >
            Whether you're building your first stock portfolio or managing an institutional fund, QuantPort India empowers you with the same advanced models used by global quant funds—including <strong>Mean-Variance Optimization</strong>, <strong>CVaR</strong>, <strong>HERC</strong>, <strong>NCO</strong>, and more. Our platform is engineered to maximize returns, control downside risk, and adapt to the unique characteristics of NSE/BSE-listed stocks.
          </Typography>
          
          <Box 
            sx={{ 
              bgcolor: 'rgba(0, 82, 204, 0.03)', 
              p: 3, 
              borderRadius: 2,
              mb: 4,
              border: '1px solid rgba(0, 82, 204, 0.1)'
            }}
          >
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 600, 
                mb: 2,
                color: '#0052cc'
              }}
            >
              Key features that set us apart:
            </Typography>
            
            <ul style={{ 
              marginLeft: 8, 
              marginTop: 0,
              marginBottom: 0,
              paddingLeft: 16,
              listStyleType: 'none'
            }}>
              <li style={{ 
                position: 'relative',
                paddingLeft: 24,
                marginBottom: 16,
                fontSize: '1.02rem',
                lineHeight: 1.6
              }}>
                <span style={{ 
                  position: 'absolute',
                  left: 0,
                  top: '0.3rem',
                  width: 8,
                  height: 8,
                  backgroundColor: '#0052cc',
                  borderRadius: '50%'
                }}></span>
                <strong>Indian market focus:</strong> Supports NIFTY, SENSEX, and BANK NIFTY as benchmarks, with dynamic risk-free rate calculations based on real 10Y G-Sec yields.
              </li>
              <li style={{ 
                position: 'relative',
                paddingLeft: 24,
                marginBottom: 16,
                fontSize: '1.02rem',
                lineHeight: 1.6
              }}>
                <span style={{ 
                  position: 'absolute',
                  left: 0,
                  top: '0.3rem',
                  width: 8,
                  height: 8,
                  backgroundColor: '#0052cc',
                  borderRadius: '50%'
                }}></span>
                <strong>Transparent, actionable analytics:</strong> Rolling betas, Sharpe/Sortino ratios, drawdowns, VaR, CVaR, and more—delivered in clear, downloadable reports.
              </li>
              <li style={{ 
                position: 'relative',
                paddingLeft: 24,
                marginBottom: 16,
                fontSize: '1.02rem',
                lineHeight: 1.6
              }}>
                <span style={{ 
                  position: 'absolute',
                  left: 0,
                  top: '0.3rem',
                  width: 8,
                  height: 8,
                  backgroundColor: '#0052cc',
                  borderRadius: '50%'
                }}></span>
                <strong>For all investors:</strong> Simple for beginners, powerful for quants and asset managers—customize constraints, target returns, and risk metrics to fit any strategy.
              </li>
              <li style={{ 
                position: 'relative',
                paddingLeft: 24,
                marginBottom: 16,
                fontSize: '1.02rem',
                lineHeight: 1.6
              }}>
                <span style={{ 
                  position: 'absolute',
                  left: 0,
                  top: '0.3rem',
                  width: 8,
                  height: 8,
                  backgroundColor: '#0052cc',
                  borderRadius: '50%'
                }}></span>
                <strong>Real-time data integration:</strong> Pulls up-to-date stock prices and market data directly from NSE/BSE for maximum reliability.
              </li>
              <li style={{ 
                position: 'relative',
                paddingLeft: 24,
                marginBottom: 0,
                fontSize: '1.02rem',
                lineHeight: 1.6
              }}>
                <span style={{ 
                  position: 'absolute',
                  left: 0,
                  top: '0.3rem',
                  width: 8,
                  height: 8,
                  backgroundColor: '#0052cc',
                  borderRadius: '50%'
                }}></span>
                <strong>Secure and privacy-first:</strong> No account needed, no personal data sold or shared. Your analytics and simulations stay confidential.
              </li>
            </ul>
          </Box>
          
          <Typography 
            variant="body1" 
            sx={{ 
              mb: 3,
              fontSize: '1.05rem',
              lineHeight: 1.7
            }}
          >
            Built by a team of quantitative finance experts, engineers, and Indian market specialists, QuantPort India's mission is to democratize advanced, research-backed portfolio optimization—bringing institutional-grade analytics to every investor.
          </Typography>
          
          <Typography 
            variant="body1"
            sx={{
              fontSize: '1.05rem',
              lineHeight: 1.7,
              fontWeight: 500,
              color: '#1e293b'
            }}
          >
            Join thousands of Indian investors, advisors, and fund managers using QuantPort India to make smarter, data-driven decisions in an ever-evolving market.
          </Typography>
        </Paper>
      </Box>
    </>
  );
} 