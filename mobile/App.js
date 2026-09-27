import React, { useState } from 'react';
import {
  SafeAreaView, ScrollView, View, Text, TextInput,
  TouchableOpacity, ActivityIndicator, StyleSheet,
} from 'react-native';
import * as Clipboard from 'expo-clipboard';

// Change this to the address that server.py prints when it starts.
//   Same Wi-Fi:  http://192.168.1.50:8000
//   Anywhere:    http://100.x.x.x:8000   (example Tailscale address)
// You can also edit it in the app itself.
const DEFAULT_URL = 'http://192.168.1.50:8000';

export default function App() {
  const [url, setUrl] = useState(DEFAULT_URL);
  const [tab, setTab] = useState('day');            // 'day' | 'long'
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Pick a tab, set your laptop address, then tap a scan.');
  const [text, setText] = useState('');

  async function copyAll() {
    if (!text) {
      setStatus('Nothing to copy yet - run a scan first.');
      return;
    }
    try {
      await Clipboard.setStringAsync(text);
      setStatus('Copied to clipboard. Paste into the Claude app.');
    } catch (e) {
      setStatus('Copy failed: ' + e.message);
    }
  }

  async function run(view) {
    const base = url.trim().replace(/\/+$/, '');
    if (!base.startsWith('http')) {
      setStatus('Enter a valid address like http://192.168.1.50:8000');
      return;
    }
    setLoading(true);
    setStatus('Scanning (' + view + ')... this takes ~30-60s');
    setText('');

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 90000);
    try {
      const res = await fetch(base + '/scan?view=' + view, { signal: controller.signal });
      const data = await res.json();
      if (!data.ok) {
        setStatus('Error: ' + (data.error || 'scan failed'));
        return;
      }
      setText(data.text);
      await Clipboard.setStringAsync(data.text);
      setStatus('Copied to clipboard. ' + data.scanned_at + ' - ' + data.summary);
    } catch (e) {
      setStatus('Could not reach the scanner. Check the address and that '
        + 'server.py is running on your laptop. (' + e.message + ')');
    } finally {
      clearTimeout(timer);
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
        <Text style={styles.h1}>Trading Scan</Text>

        <Text style={styles.label}>Laptop address (from server.py)</Text>
        <TextInput
          style={styles.input}
          value={url}
          onChangeText={setUrl}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="http://192.168.1.50:8000"
          placeholderTextColor="#7c8896"
        />

        <View style={styles.tabs}>
          <Tab label="Day Trading" active={tab === 'day'} onPress={() => setTab('day')} />
          <Tab label="Long Term" active={tab === 'long'} onPress={() => setTab('long')} />
        </View>

        {tab === 'day' ? (
          <View>
            <Text style={styles.hint}>Intraday and short holds - in and out same day or a few days.</Text>
            <View style={styles.row}>
              <Btn label="Buy now" onPress={() => run('buy')} disabled={loading} />
              <Btn label="Morning hold" onPress={() => run('morning')} disabled={loading} />
            </View>
            <View style={styles.row}>
              <Btn label="Overnight" onPress={() => run('overnight')} disabled={loading} />
              <Btn label="Swing (days)" onPress={() => run('swing')} disabled={loading} />
            </View>
            <Btn label="All day-trading views" onPress={() => run('day')} disabled={loading} alt wide />
          </View>
        ) : (
          <View>
            <Text style={styles.hint}>Held for weeks - uptrend pullbacks and tracking what you own.</Text>
            <Btn label="Long hold (weeks)" onPress={() => run('position')} disabled={loading} wide />
            <View style={{ height: 10 }} />
            <Btn label="My positions" onPress={() => run('holdings')} disabled={loading} alt wide />
          </View>
        )}

        <View style={{ height: 10 }} />
        <Btn label="Everything (both)" onPress={() => run('all')} disabled={loading} alt wide />

        {loading ? <ActivityIndicator style={{ marginVertical: 12 }} color="#fff" /> : null}

        <TouchableOpacity
          onPress={copyAll}
          disabled={!text}
          style={[styles.copyBtn, !text && styles.btnDisabled]}
        >
          <Text style={styles.btnText}>Copy all to clipboard</Text>
        </TouchableOpacity>

        <Text style={styles.status}>{status}</Text>
        {text ? <Text selectable style={styles.output}>{text}</Text> : null}
      </ScrollView>
    </SafeAreaView>
  );
}

function Tab({ label, active, onPress }) {
  return (
    <TouchableOpacity onPress={onPress} style={[styles.tab, active && styles.tabActive]}>
      <Text style={[styles.tabText, active && styles.tabTextActive]}>{label}</Text>
    </TouchableOpacity>
  );
}

function Btn({ label, onPress, disabled, alt, wide }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={disabled}
      style={[styles.btn, wide && styles.btnWide, alt && styles.btnAlt, disabled && styles.btnDisabled]}
    >
      <Text style={styles.btnText}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0b0f14' },
  container: { padding: 16 },
  h1: { color: '#ffffff', fontSize: 22, fontWeight: '600', marginBottom: 14 },
  label: { color: '#9aa6b2', fontSize: 13, marginBottom: 4 },
  input: {
    backgroundColor: '#161c24', color: '#ffffff', borderRadius: 10,
    padding: 12, fontSize: 15, marginBottom: 14,
  },
  tabs: { flexDirection: 'row', marginBottom: 12 },
  tab: {
    flex: 1, paddingVertical: 12, alignItems: 'center', borderRadius: 12,
    borderWidth: 2, borderColor: '#4b5563', marginHorizontal: 4,
  },
  tabActive: { backgroundColor: '#2f6df6', borderColor: '#2f6df6' },
  tabText: { color: '#9aa6b2', fontSize: 15, fontWeight: '600' },
  tabTextActive: { color: '#ffffff' },
  hint: { color: '#7c8896', fontSize: 12, marginBottom: 10, marginHorizontal: 2 },
  row: { flexDirection: 'row', marginBottom: 10 },
  btn: {
    flex: 1, backgroundColor: '#2f6df6', borderRadius: 12,
    paddingVertical: 16, alignItems: 'center', marginHorizontal: 5,
  },
  btnWide: { marginHorizontal: 5 },
  btnAlt: { backgroundColor: '#4b5563' },
  btnDisabled: { opacity: 0.5 },
  btnText: { color: '#ffffff', fontSize: 16, fontWeight: '600' },
  copyBtn: {
    backgroundColor: '#1f9d55', borderRadius: 12, paddingVertical: 16,
    alignItems: 'center', marginTop: 6, marginBottom: 4,
  },
  status: { color: '#cdd5df', fontSize: 14, marginVertical: 10 },
  output: { color: '#e6edf3', fontFamily: 'Courier', fontSize: 12, marginTop: 8 },
});
