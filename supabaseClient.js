const { createClient } = require('@supabase/supabase-js');

// Use your real Supabase URL and service key
const supabaseUrl = 'https://tyfttcxihduohajlzmfn.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR5ZnR0Y3hpaGR1b2hhamx6bWZuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MzY5NzUyNiwiZXhwIjoyMDU5MjczNTI2fQ.QSUX_9W8G-bFqEUqgI07b8B0eqlTMmi6ws9CGlRgW6k';

const supabase = createClient(supabaseUrl, supabaseKey);

module.exports = supabase;