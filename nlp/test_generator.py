"""
nlp/test_generator.py
Generates bilingual Unit + ECU Integration test cases from BMS requirements.
"""

import re
import pandas as pd
from loguru import logger


class BMSTestCaseGenerator:
    """Generates structured bilingual test cases from NLP-enriched requirements."""

    # ── Unit Test Cases ───────────────────────────────────────────────

    def generate_unit_tests(self, row: pd.Series) -> list[dict]:
        req_id   = row["Item ID"]
        section  = str(row["Gliederungsnummer"])
        text_en  = row["text_en"]
        text_de  = row["text_de"]
        ents     = row["bms_entities"]
        is_crit  = row["is_critical"]
        topic_en = row.get("topic_label_en", "")
        topic_de = row.get("topic_label_de", "")
        base     = f"TC_{section.replace('.', '_')}"
        tcs      = []

        # 1. Nominal
        tcs.append({
            "TC_ID":              f"{base}_001",
            "Requirement_ID":     req_id,
            "Section":            section,
            "Topic_EN":           topic_en,
            "Topic_DE":           topic_de,
            "Category":           "Unit",
            "Type_EN":            "Nominal",
            "Type_DE":            "Normalfall",
            "Objective_EN":       f"Verify nominal output for requirement {section}",
            "Ziel_DE":            f"Nominalverhalten für Anforderung {section} prüfen",
            "Preconditions_EN":   "All qualifiers = QUAL_INT_OK; inputs within valid range",
            "Vorbedingungen_DE":  "Alle Qualifier = QUAL_INT_OK; Eingangswerte im gültigen Bereich",
            "Inputs":             ", ".join(ents["signals"][:3]) or "N/A",
            "Qualifiers":         ", ".join(ents["qualifiers"]) or "N/A",
            "Thresholds_C":       ", ".join(ents["thresholds"]) or "N/A",
            "Expected_Result_EN": "Output equals input when plausibility conditions are met",
            "Erwartetes_Ergebnis_DE": "Ausgangssignal entspricht Eingangswert bei erfüllten Plausibilitätsbedingungen",
            "Priority":           "High" if is_crit else "Medium",
            "Criticality_Score":  row["criticality_score"],
            "Test_Environment":   "SIL / Software unit test",
        })

        # 2. Boundary per threshold
        for i, thr in enumerate(ents["thresholds"][:2]):
            tcs.append({
                "TC_ID":              f"{base}_{10+i:03d}",
                "Requirement_ID":     req_id,
                "Section":            section,
                "Topic_EN":           topic_en,
                "Topic_DE":           topic_de,
                "Category":           "Unit",
                "Type_EN":            "Boundary",
                "Type_DE":            "Grenzwert",
                "Objective_EN":       f"Verify behavior at boundary {thr}°C",
                "Ziel_DE":            f"Verhalten an Grenzwert {thr}°C prüfen",
                "Preconditions_EN":   f"Input = {thr}°C (boundary condition)",
                "Vorbedingungen_DE":  f"Eingangswert = {thr}°C (Grenzwert)",
                "Inputs":             ", ".join(ents["signals"][:2]) or "N/A",
                "Qualifiers":         ", ".join(ents["qualifiers"]) or "N/A",
                "Thresholds_C":       thr,
                "Expected_Result_EN": f"Output correctly handled at boundary {thr}°C",
                "Erwartetes_Ergebnis_DE": f"Ausgangswert korrekt am Grenzwert {thr}°C",
                "Priority":           "High",
                "Criticality_Score":  row["criticality_score"],
                "Test_Environment":   "SIL / Software unit test",
            })

        # 3. Fault
        fault_kw_en = ["plausib", "qual_int", "qualifier", "invalid"]
        fault_kw_de = ["plausib", "qualifier", "ungültig", "n.i.o", "fehler"]
        if any(k in text_en.lower() for k in fault_kw_en) or \
           any(k in text_de.lower() for k in fault_kw_de):
            tcs.append({
                "TC_ID":              f"{base}_099",
                "Requirement_ID":     req_id,
                "Section":            section,
                "Topic_EN":           topic_en,
                "Topic_DE":           topic_de,
                "Category":           "Unit",
                "Type_EN":            "Fault",
                "Type_DE":            "Fehlerfall",
                "Objective_EN":       "Verify behavior when qualifier is invalid / not OK",
                "Ziel_DE":            "Verhalten bei ungültigem Qualifier prüfen",
                "Preconditions_EN":   "Qualifier ≠ QUAL_INT_OK (e.g. QUAL_INT_INIT)",
                "Vorbedingungen_DE":  "Qualifier ≠ QUAL_INT_OK (z.B. QUAL_INT_INIT oder n.i.O.)",
                "Inputs":             ", ".join(ents["qualifiers"]) or "N/A",
                "Qualifiers":         ", ".join(ents["qualifiers"]) or "N/A",
                "Thresholds_C":       "N/A",
                "Expected_Result_EN": "Output not updated; safe/default value applied",
                "Erwartetes_Ergebnis_DE": "Ausgangssignal nicht aktualisiert; Standardwert gesetzt",
                "Priority":           "Critical" if is_crit else "High",
                "Criticality_Score":  row["criticality_score"],
                "Test_Environment":   "SIL / Software unit test",
            })

        # 4. Out-of-Range
        if len(ents["thresholds"]) >= 1:
            tcs.append({
                "TC_ID":              f"{base}_098",
                "Requirement_ID":     req_id,
                "Section":            section,
                "Topic_EN":           topic_en,
                "Topic_DE":           topic_de,
                "Category":           "Unit",
                "Type_EN":            "Out-of-Range",
                "Type_DE":            "Bereichsüberschreitung",
                "Objective_EN":       "Verify behavior when input is outside valid range",
                "Ziel_DE":            "Verhalten bei Eingangswert außerhalb des gültigen Bereichs prüfen",
                "Preconditions_EN":   f"Input < {ents['thresholds'][0]}°C OR > {ents['thresholds'][-1]}°C",
                "Vorbedingungen_DE":  f"Eingangswert < {ents['thresholds'][0]}°C ODER > {ents['thresholds'][-1]}°C",
                "Inputs":             ", ".join(ents["signals"][:2]) or "N/A",
                "Qualifiers":         ", ".join(ents["qualifiers"]) or "N/A",
                "Thresholds_C":       f"{ents['thresholds'][0]} / {ents['thresholds'][-1]}",
                "Expected_Result_EN": "Output blocked; error/default value set",
                "Erwartetes_Ergebnis_DE": "Ausgangswert gesperrt; Fehlerwert gesetzt",
                "Priority":           "High",
                "Criticality_Score":  row["criticality_score"],
                "Test_Environment":   "SIL / Software unit test",
            })

        return tcs

    # ── ECU Integration Test Cases ────────────────────────────────────

    def generate_ecu_tests(self, row: pd.Series) -> list[dict]:
        req_id    = row["Item ID"]
        section   = str(row["Gliederungsnummer"])
        text_en   = row["text_en"]
        text_de   = row["text_de"]
        ents      = row["bms_entities"]
        ecu_level = row.get("ecu_level", "Software Unit")
        is_crit   = row["is_critical"]
        base      = f"ECU_TC_{section.replace('.', '_')}"
        text_all  = (text_en + " " + text_de).lower()
        tcs       = []

        # 1. End-to-End (always)
        tcs.append({
            "TC_ID":                  f"{base}_E2E_001",
            "Requirement_ID":         req_id,
            "Section":                section,
            "ECU_Level":              ecu_level,
            "Category":               "ECU Integration",
            "Integration_Type_EN":    "End-to-End Signal Flow",
            "Integration_Type_DE":    "Ende-zu-Ende Signalfluss",
            "Test_Environment":       "HIL (Hardware-in-the-Loop)",
            "Objective_EN":           f"Verify complete signal path from sensor input to ECU output for req {section}",
            "Ziel_DE":                f"Vollständigen Signalweg von Sensor bis ECU-Ausgang für Anforderung {section} prüfen",
            "Preconditions_EN":       "ECU ON; HIL bench connected; all CAN nodes active; ignition ON",
            "Vorbedingungen_DE":      "ECU eingeschaltet; HIL-Prüfstand verbunden; alle CAN-Knoten aktiv; Zündung EIN",
            "Stimulus_Signals":       ", ".join(ents["signals"][:3]) or "N/A",
            "Expected_Result_EN":     "Output correctly propagated within cycle time; no CAN error frames",
            "Erwartetes_Ergebnis_DE": "Ausgangssignal korrekt übertragen innerhalb Zykluszeit; keine CAN-Fehlerframes",
            "Measurement_Tool":       "CANalyzer / CANoe / dSPACE ControlDesk",
            "Priority":               "Critical" if is_crit else "High",
            "Criticality_Score":      row["criticality_score"],
        })

        # 2. OBD / Diagnostics
        if re.search(r"\b(OBD|ITID|PID|diagnostic|DTC|UDS)\b", text_all, re.IGNORECASE):
            pid_matches = re.findall(r"PID[-_]?([0-9A-Fa-f]{4})", text_en + text_de)
            pid_str = ", ".join(set(pid_matches)) if pid_matches else "OBD PID (see requirement)"
            tcs.append({
                "TC_ID":                  f"{base}_OBD_001",
                "Requirement_ID":         req_id,
                "Section":                section,
                "ECU_Level":              ecu_level,
                "Category":               "ECU Integration",
                "Integration_Type_EN":    "OBD / Diagnostic Interface",
                "Integration_Type_DE":    "OBD / Diagnoseschnittstelle",
                "Test_Environment":       "HIL + Diagnostic Tester (ISTA / CANoe DiagVistA)",
                "Objective_EN":           f"Verify OBD/diagnostic data and PID response for {pid_str}",
                "Ziel_DE":                f"OBD-Datenverfügbarkeit und PID-Antwort prüfen für {pid_str}",
                "Preconditions_EN":       "OBD tester connected via OBD-II port; ECU in diagnostic session",
                "Vorbedingungen_DE":      "OBD-Tester über OBD-II verbunden; ECU in Diagnosesitzung",
                "Stimulus_Signals":       pid_str,
                "Expected_Result_EN":     "Correct PID response within 50ms; data matches ECU internal value",
                "Erwartetes_Ergebnis_DE": "Korrekte PID-Antwort innerhalb 50ms; Daten stimmen mit ECU-Wert überein",
                "Measurement_Tool":       "ISTA / CANoe DiagVistA / python-udsoncan",
                "Priority":               "High",
                "Criticality_Score":      row["criticality_score"],
            })

        # 3. CAN Bus
        if ents["signals"] or re.search(r"\b(CAN|LIN|PDU|bus|frame)\b", text_all, re.IGNORECASE):
            tcs.append({
                "TC_ID":                  f"{base}_CAN_001",
                "Requirement_ID":         req_id,
                "Section":                section,
                "ECU_Level":              ecu_level,
                "Category":               "ECU Integration",
                "Integration_Type_EN":    "CAN Bus Signal Verification",
                "Integration_Type_DE":    "CAN-Bus Signalverifikation",
                "Test_Environment":       "HIL + CANalyzer / CANoe",
                "Objective_EN":           f"Verify CAN transmission of: {', '.join(ents['signals'][:2]) or 'output signals'}",
                "Ziel_DE":                f"CAN-Übertragung prüfen von: {', '.join(ents['signals'][:2]) or 'Ausgangssignale'}",
                "Preconditions_EN":       "HIL bench active; CAN network nominal; ECU awake",
                "Vorbedingungen_DE":      "HIL-Prüfstand aktiv; CAN-Netzwerk nominal; ECU aktiv",
                "Stimulus_Signals":       ", ".join(ents["signals"][:3]) or "N/A",
                "Expected_Result_EN":     "Signal on CAN with correct ID, DLC, value, cycle time; no bus-off",
                "Erwartetes_Ergebnis_DE": "Signal auf CAN mit korrekter ID, DLC, Wert und Zykluszeit; kein Bus-Off",
                "Measurement_Tool":       "CANalyzer / CANoe / python-can",
                "Priority":               "High",
                "Criticality_Score":      row["criticality_score"],
            })

        # 4. HIL Fault Injection
        if is_crit or re.search(r"\b(plausib|qualifier|QUAL_INT|n\.i\.o|fehler|fault|invalid)\b",
                                 text_all, re.IGNORECASE):
            tcs.append({
                "TC_ID":                  f"{base}_HIL_001",
                "Requirement_ID":         req_id,
                "Section":                section,
                "ECU_Level":              ecu_level,
                "Category":               "ECU Integration",
                "Integration_Type_EN":    "HIL Fault Injection",
                "Integration_Type_DE":    "HIL Fehlerinjektion",
                "Test_Environment":       "HIL — dSPACE / NI VeriStand",
                "Objective_EN":           "Verify ECU response when sensor/qualifier becomes invalid on HIL bench",
                "Ziel_DE":                "ECU-Fehlerreaktion prüfen wenn Sensor/Qualifier am HIL ungültig wird",
                "Preconditions_EN":       "HIL bench nominal; inject: wire break / short-circuit / out-of-range",
                "Vorbedingungen_DE":      "HIL-Prüfstand nominal; Injektion: Leitungsunterbrechung / Kurzschluss",
                "Stimulus_Signals":       ", ".join(ents["signals"][:2]) or "N/A",
                "Expected_Result_EN":     "ECU sets qualifier invalid; output = error/default value; DTC stored",
                "Erwartetes_Ergebnis_DE": "ECU setzt Qualifier ungültig; Ausgang = Fehlerwert; DTC gespeichert",
                "Measurement_Tool":       "dSPACE ControlDesk / CANoe / INCA",
                "Priority":               "Critical" if is_crit else "High",
                "Criticality_Score":      row["criticality_score"],
            })

        # 5. Timing
        if re.search(r"\b(cycle|zykl|task|ms|period|10ms|20ms|100ms)\b", text_all, re.IGNORECASE):
            cycle_matches = re.findall(r"(\d+)\s*ms", text_en + text_de)
            cycle_str = f"{cycle_matches[0]} ms" if cycle_matches else "defined cycle time"
            tcs.append({
                "TC_ID":                  f"{base}_TIMING_001",
                "Requirement_ID":         req_id,
                "Section":                section,
                "ECU_Level":              ecu_level,
                "Category":               "ECU Integration",
                "Integration_Type_EN":    "Timing & Cycle Verification",
                "Integration_Type_DE":    "Timing & Zyklusverifikation",
                "Test_Environment":       "HIL + CANoe Measurement / Oscilloscope",
                "Objective_EN":           f"Verify signal update within cycle time ({cycle_str})",
                "Ziel_DE":                f"Signalaktualisierung innerhalb Zykluszeit ({cycle_str}) prüfen",
                "Preconditions_EN":       "ECU running; timer capturing signal transitions",
                "Vorbedingungen_DE":      "ECU läuft; Timer erfasst Signalübergänge",
                "Stimulus_Signals":       ", ".join(ents["signals"][:2]) or "N/A",
                "Expected_Result_EN":     f"Update interval ≤ {cycle_str}; jitter within ±10%",
                "Erwartetes_Ergebnis_DE": f"Aktualisierungsintervall ≤ {cycle_str}; Jitter innerhalb ±10%",
                "Measurement_Tool":       "CANoe / Oscilloscope / dSPACE ControlDesk",
                "Priority":               "High",
                "Criticality_Score":      row["criticality_score"],
            })

        # 6. DTC / Error Memory
        if re.search(r"\b(DTC|error memory|Fehlerspeicher|freeze frame|DFCC)\b",
                      text_all, re.IGNORECASE):
            tcs.append({
                "TC_ID":                  f"{base}_DTC_001",
                "Requirement_ID":         req_id,
                "Section":                section,
                "ECU_Level":              ecu_level,
                "Category":               "ECU Integration",
                "Integration_Type_EN":    "Error Memory / DTC Verification",
                "Integration_Type_DE":    "Fehlerspeicher / DTC Verifikation",
                "Test_Environment":       "HIL + Diagnostic Tester (ISTA / CANoe DiagVistA)",
                "Objective_EN":           "Verify DTC stored in ECU error memory when fault condition active",
                "Ziel_DE":                "Prüfen ob DTC im ECU-Fehlerspeicher gespeichert wenn Fehlerbedingung aktiv",
                "Preconditions_EN":       "Inject fault; read DTC via UDS service 0x19",
                "Vorbedingungen_DE":      "Fehlerbedingung injizieren; DTC via UDS-Dienst 0x19 auslesen",
                "Stimulus_Signals":       ", ".join(ents["signals"][:2]) or "N/A",
                "Expected_Result_EN":     "DTC stored with status 0x09; freeze frame data available",
                "Erwartetes_Ergebnis_DE": "DTC gespeichert mit Statusbyte 0x09; Freeze-Frame-Daten verfügbar",
                "Measurement_Tool":       "ISTA / python-udsoncan / CANoe DiagVistA",
                "Priority":               "High",
                "Criticality_Score":      row["criticality_score"],
            })

        return tcs

    # ── Full Run ──────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate all unit + ECU test cases.
        Returns (df_unit_tcs, df_ecu_tcs)
        """
        logger.info("Generating unit test cases...")
        unit_tcs = [tc for _, row in df.iterrows() for tc in self.generate_unit_tests(row)]
        df_unit = pd.DataFrame(unit_tcs)
        logger.info(f"  Unit TCs: {len(df_unit)}")

        logger.info("Generating ECU integration test cases...")
        ecu_tcs = [tc for _, row in df.iterrows() for tc in self.generate_ecu_tests(row)]
        df_ecu = pd.DataFrame(ecu_tcs)
        logger.info(f"  ECU TCs: {len(df_ecu)}")

        logger.info(f"Total test cases: {len(df_unit) + len(df_ecu)} ✅")
        return df_unit, df_ecu
