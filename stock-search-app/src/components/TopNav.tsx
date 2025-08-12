import React, { useState } from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import MenuIcon from '@mui/icons-material/Menu';
import Box from '@mui/material/Box';
import Link from 'next/link';

export default function TopNav() {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleMenu = (event: React.MouseEvent<HTMLButtonElement>) => setAnchorEl(event.currentTarget);
  const handleClose = () => setAnchorEl(null);

  return (
    <AppBar position="static" color="default" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
      <Toolbar>
        <Typography variant="h6" color="primary" sx={{ flexGrow: 1, fontWeight: 700, letterSpacing: '0.03em' }}>
          <Link href="/" style={{ textDecoration: 'none', color: 'inherit', fontWeight: 800 }}>
            QuantPort India
          </Link>
        </Typography>
        {/* Hamburger on small screens */}
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ display: 'block' }}>
            <IconButton edge="end" color="inherit" aria-label="menu" onClick={handleMenu} sx={{ display: { md: 'none' } }}>
              <MenuIcon />
            </IconButton>
            <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleClose}>
              <MenuItem onClick={handleClose}>
                <Link href="/dividend" style={{ color: 'inherit', textDecoration: 'none' }}>Dividend Optimizer</Link>
              </MenuItem>
              <MenuItem onClick={handleClose}>
                <Link href="/docs" style={{ color: 'inherit', textDecoration: 'none' }}>Docs</Link>
              </MenuItem>
              <MenuItem onClick={handleClose}>
                <Link href="/about" style={{ color: 'inherit', textDecoration: 'none' }}>About</Link>
              </MenuItem>
            </Menu>
          </div>
          {/* Desktop menu */}
          <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: '32px' }}>
            <Link href="/dividend" style={{ color: '#0052cc', textDecoration: 'none', fontWeight: 500, fontSize: '1.08rem' }}>
              Dividend Optimizer
            </Link>
            <Link href="/docs" style={{ color: '#0052cc', textDecoration: 'none', fontWeight: 500, fontSize: '1.08rem' }}>
              Docs
            </Link>
            <Link href="/about" style={{ color: '#0052cc', textDecoration: 'none', fontWeight: 500, fontSize: '1.08rem' }}>
              About
            </Link>
          </Box>
        </div>
      </Toolbar>
    </AppBar>
  );
} 